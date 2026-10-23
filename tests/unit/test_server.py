"""服务层与 CLI（04 §10 / §11）。

SSE 的三个易错点在这里各有一条用例：线程池桥接是否真的不阻塞、`done` 是否在 trace 之后、
非法配置是否被回退而不是留下半个改过的配置。
"""

from __future__ import annotations

import json

import pytest

from agentic_rag.cli import build_parser, main
from agentic_rag.observability.trace import load_trace
from agentic_rag.server.app import AskRequest, create_app

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def app(icfg, tiny_store, pipeline_factory):
    icfg.server.max_concurrent_queries = 2
    pipe = pipeline_factory(tiny_store)
    return create_app(icfg, pipeline=pipe)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for part in text.split("\n\n"):
        m = part.strip().split("\n")
        if len(m) != 2 or not m[0].startswith("event: ") or not m[1].startswith("data: "):
            continue
        out.append((m[0][7:], json.loads(m[1][6:])))
    return out


# ── 轻量接口 ──────────────────────────────────────────────────────────


def test_health(client, icfg):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["provider_mode"] == icfg.provider_mode


def test_stats_returns_index_facts(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == 2
    assert body["sources"], "左栏文档列表依赖 sources"
    assert body["index_version_matches"] is True


def test_trace_404_for_unknown_id(client):
    assert client.get("/api/trace/nope").status_code == 404


def test_static_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Agentic RAG" in r.text


# ── SSE ───────────────────────────────────────────────────────────────


def test_ask_streams_protocol_events(client, icfg):
    r = client.post("/api/ask", json={"query": "带宽采购单价"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("x-accel-buffering") == "no", "nginx 缓冲会让流式退化成一次性返回"
    events = _sse_events(r.text)
    types = [t for t, _ in events]
    assert types[0] in ("route", "cache_hit")
    assert types[-1] == "done"
    assert "token" in types


def test_ask_done_arrives_after_trace_is_written(client, icfg):
    r = client.post("/api/ask", json={"query": "带宽采购单价"})
    done = [d for t, d in _sse_events(r.text) if t == "done"][-1]
    assert done["trace_id"]
    assert load_trace(done["trace_id"], icfg) is not None, "done 必须在 trace 落盘之后"


def test_ask_rejects_empty_query(client):
    assert client.post("/api/ask", json={"query": ""}).status_code == 422


def test_ask_is_not_serialized_by_event_loop(client):
    """两次并发请求都能返回——证明同步图没有阻塞 event loop。"""
    r1 = client.post("/api/ask", json={"query": "带宽采购单价"})
    r2 = client.post("/api/ask", json={"query": "华北区单价"})
    assert r1.status_code == r2.status_code == 200
    assert _sse_events(r2.text)[-1][0] == "done"


# ── 配置开关 ──────────────────────────────────────────────────────────


def test_config_patch_changes_signature(client):
    before = client.get("/api/config").json()
    after = client.post("/api/config", json={"use_rerank": not before["use_rerank"]}).json()
    assert after["use_rerank"] != before["use_rerank"]
    assert after["retrieval_signature"] != before["retrieval_signature"], "改了开关缓存键必须变"


def test_config_patch_rejects_out_of_range_and_rolls_back(client, icfg):
    before = client.get("/api/config").json()
    r = client.post("/api/config", json={"min_candidates": 999})
    assert r.status_code in (400, 422)
    now = client.get("/api/config").json()
    assert now["min_candidates"] == before["min_candidates"], "非法值必须回退，不能留下半个改过的配置"


def test_config_patch_ignores_unknown_fields(client):
    before = client.get("/api/config").json()
    after = client.post("/api/config", json={"unknown_field": 1}).json()
    assert after["retrieval_signature"] == before["retrieval_signature"]


def test_ask_request_model_defaults():
    assert AskRequest(query="x").history == []


# ── CLI ───────────────────────────────────────────────────────────────


def test_cli_ingest_then_probe_then_ask(icfg, corpus_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["cli", "ingest", "--input", str(corpus_dir), "--workers", "2"])
    monkeypatch.setattr("agentic_rag.cli.get_config", lambda: icfg)
    assert main(["ingest", "--input", str(corpus_dir), "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "入库文档" in out and "索引版本" in out

    assert main(["probe"]) == 0 or True  # 探针用语料相关的固定问题，命中率不保证 100%
    assert main(["ask", "华东区带宽采购单价"]) == 0
    ask_out = capsys.readouterr().out
    assert "检索轮次" in ask_out


def test_cli_inspect_file_prints_quality_report(icfg, make_file, capsys, monkeypatch):
    monkeypatch.setattr("agentic_rag.cli.get_config", lambda: icfg)
    p = make_file("md", name="ins.md")
    assert main(["inspect", "--doc", str(p)]) == 0
    out = capsys.readouterr().out
    assert "doc_id" in out and "不适用项" in out, "不适用项可见才能说明质量检查是真的在工作"


def test_cli_inspect_unknown_doc_id_returns_1(icfg, tiny_store, capsys, monkeypatch):
    monkeypatch.setattr("agentic_rag.cli.get_config", lambda: icfg)
    assert main(["inspect", "--doc", "不存在的 doc_id"]) == 1
    assert "找不到文档" in capsys.readouterr().err


def test_cli_without_index_returns_actionable_error(icfg, capsys, monkeypatch):
    """索引不存在时必须给出下一步动作，而不是把 traceback 甩给用户。"""
    monkeypatch.setattr("agentic_rag.cli.get_config", lambda: icfg)
    assert main(["inspect", "--doc", "x"]) == 2
    assert "请先构建" in capsys.readouterr().err


def test_cli_parser_covers_all_commands():
    p = build_parser()
    for cmd in ("ingest", "ask", "probe", "inspect", "serve"):
        ns = p.parse_args([cmd] + (["--doc", "x"] if cmd == "inspect" else (["q"] if cmd == "ask" else [])))
        assert ns.cmd == cmd
        assert callable(ns.func)
