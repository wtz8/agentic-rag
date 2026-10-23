"""索引层：BM25 手算校验、向量库一致性、版本校验（07 §2 / §3）。"""

from __future__ import annotations

import math

import pytest

from agentic_rag.index.bm25 import BM25Index, tokenize
from agentic_rag.index.store import IndexStore, IndexVersionMismatch
from agentic_rag.index.vector_store import (
    COSINE,
    IndexLocked,
    InMemoryStore,
    QdrantStore,
    get_vector_store,
    to_point_id,
)
from agentic_rag.schema import Chunk

DOCS = [
    ("d0", "华东区 带宽 采购 单价 3.2 万元"),
    ("d1", "华北区 带宽 采购 单价 2.8 万元"),
    ("d2", "设备 型号 XR-2400 支持 双电源 冗余"),
]


def _mk(cid: str, text: str) -> Chunk:
    c = Chunk(chunk_id=cid, doc_id=cid, parent_id=None, raw_text=text, text=text, source="t.md")
    c.tokens = len(text)
    return c


def _mini_index(k1: float = 1.5, b: float = 0.75) -> BM25Index:
    idx = BM25Index(k1=k1, b=b)
    idx.build([_mk(cid, t) for cid, t in DOCS])
    return idx


# ── BM25 手算对照 ─────────────────────────────────────────────────────


def test_bm25_hand_computed_score():
    """手算对抗：IDF 与长度归一化的符号最容易写反。"""
    k1, b = 1.5, 0.75
    idx = _mini_index(k1, b)
    docs_tokens = [tokenize(t) for _, t in DOCS]
    n = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / n
    df = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    query = ["带宽", "采购"]
    expected = {}
    for i, tokens in enumerate(docs_tokens):
        score = 0.0
        dl = len(tokens)
        for q in query:
            tf = tokens.count(q)
            if tf == 0:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        expected[DOCS[i][0]] = score

    got = dict(idx.search("带宽 采购", 10))
    for cid, exp in expected.items():
        assert got.get(cid, 0.0) == pytest.approx(round(exp, 6), abs=1e-6)


def test_bm25_ranking_and_zero_for_missing_term():
    idx = _mini_index()
    hits = idx.search("华东区 带宽", 5)
    assert hits[0][0] == "d0"
    assert idx.search("完全不存在的词", 5) == []


def test_bm25_idf_direction():
    """稀有词必须比常见词权重更高（IDF 符号写反会翻转这条）。"""
    idx = BM25Index()
    idx.build([_mk("a", "带宽 带宽 带宽 常见词"), _mk("b", "带宽 罕见术语")])
    common = idx.idf["常见词"] if "常见词" in idx.idf else None
    rare = idx.idf["罕见术语"] if "罕见术语" in idx.idf else None
    if common is not None and rare is not None:
        assert rare > common


def test_bm25_length_normalization_direction():
    """同样 tf 下，短文档得分更高（长度归一化符号写反会翻转这条）。"""
    idx = BM25Index()
    short = _mk("s", "带宽 " + "噪声 " * 2)
    long_ = _mk("l", "带宽 " + "噪声 " * 60)
    idx.build([short, long_])
    scores = dict(idx.search("带宽", 2))
    assert scores["s"] > scores["l"]


def test_bm25_empty_query_and_index():
    assert _mini_index().search("", 5) == []
    assert BM25Index().search("带宽", 5) == []


def test_bm25_save_load_roundtrip(tmp_path):
    idx = _mini_index()
    p = tmp_path / "bm25.pkl"
    idx.save(p)
    loaded = BM25Index.load(p)
    assert loaded.search("带宽 采购 单价", 3) == idx.search("带宽 采购 单价", 3)
    assert loaded.stats()["chunks"] == 3


def test_tokenize_filters_punctuation():
    tokens = tokenize("华东区，带宽！单价 3.2 万元。")
    assert "带宽" in tokens
    assert all(t.strip() and not t.isspace() for t in tokens)


# ── 向量库 ────────────────────────────────────────────────────────────


def _vecs(dim: int = 8):
    import numpy as np

    rng = np.random.default_rng(0)
    return [list(rng.normal(size=dim)) for _ in range(3)]


def test_inmemory_and_qdrant_agree(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_q")
    cfg.index.backend = "qdrant"
    q = QdrantStore(cfg)
    q.create(8, "v1")
    vs = _vecs()
    q.upsert(["a", "b", "c"], vs, [{"chunk_id": i, "doc_id": "d"} for i in "abc"])
    mem = InMemoryStore(cfg)
    mem.create(8, "v1")
    mem.upsert(["a", "b", "c"], vs, [{"chunk_id": i, "doc_id": "d"} for i in "abc"])

    q_hits = q.search(vs[0], 3)
    m_hits = mem.search(vs[0], 3)
    assert [i for i, _ in q_hits] == [i for i, _ in m_hits]
    assert q_hits[0][1] == pytest.approx(m_hits[0][1], abs=1e-5)
    assert q_hits[0][0] == "a" and q_hits[0][1] == pytest.approx(1.0, abs=1e-5)  # 自身余弦 = 1


def test_point_id_is_stable_and_uuid():
    assert to_point_id("x:c0000") == to_point_id("x:c0000")
    assert len(to_point_id("x:c0000")) == 36


def test_delete_by_doc_and_count(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_m")
    cfg.index.backend = "memory"
    store = get_vector_store(cfg)
    store.create(4, "v1")
    store.upsert(["a", "b"], [[1, 0, 0, 0], [0, 1, 0, 0]], [{"doc_id": "d1"}, {"doc_id": "d2"}])
    assert store.count() == 2
    store.delete_by_doc("d1")
    assert store.count() == 1


def test_meta_roundtrip(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_m2")
    cfg.index.backend = "memory"
    store = get_vector_store(cfg)
    store.create(4, "v1")
    store.save_meta({"version": "v1", "dim": 4})
    assert store.load_meta() == {"version": "v1", "dim": 4}


def test_cosine_is_enforced_on_existing_collection(cfg, tmp_path, monkeypatch):
    """改度量会让 sufficiency_vector_floor 静默失效，必须直接报错（K2）。"""
    cfg.index.path = str(tmp_path / "idx_c")
    cfg.index.backend = "qdrant"
    store = QdrantStore(cfg)

    class _Cfg:
        class params:
            class vectors:
                distance = "Euclid"

    class _Info:
        config = _Cfg()

    monkeypatch.setattr(store._client, "collection_exists", lambda name: True)
    monkeypatch.setattr(store._client, "get_collection", lambda name: _Info())
    with pytest.raises(ValueError, match="距离度量"):
        store.create(8, "v1")
    assert COSINE == "Cosine"


def test_qdrant_local_can_be_reopened(cfg, tmp_path):
    """锁文件的存在≠被占用：Qdrant 关闭后仍保留 `.lock`，重开完全正常。

    按"文件在就是被占用"判断，结果是第二次运行 CLI 就报"被其他进程打开"——
    而那个进程根本不存在。这条用例锁住的是"能不能第二次运行"。
    """
    cfg.index.path = str(tmp_path / "idx_reopen")
    cfg.index.backend = "qdrant"
    first = QdrantStore(cfg)
    first.create(4, "v1")
    # 模拟"CLI 第一次运行结束"：客户端释放后，锁文件仍然留在磁盘上
    first._client.close()
    # Qdrant local 的数据在 `qdrant/` 子目录（meta.json 冲突的修复），锁文件也在那里
    assert (first._base / "qdrant" / ".lock").exists(), "Qdrant 关闭后不删锁文件——它却可被重开"
    second = QdrantStore(cfg)  # 不得抛 IndexLocked
    assert second.count() == first.count()


def test_qdrant_local_raises_index_locked_when_truly_held(cfg, tmp_path, monkeypatch):
    """真被占用时要给出可操作的错误，而不是悄悄降级成内存库。"""
    import qdrant_client

    cfg.index.path = str(tmp_path / "idx_locked")
    cfg.index.backend = "qdrant"

    def boom(*a, **k):
        raise RuntimeError(
            f"Storage folder {k.get('path', '')} is already accessed by another instance of Qdrant client."
        )

    monkeypatch.setattr(qdrant_client, "QdrantClient", boom)
    with pytest.raises(IndexLocked, match="已被其他进程打开"):
        QdrantStore(cfg)


def test_qdrant_server_requires_url(cfg):
    cfg.index.backend = "qdrant_server"
    cfg.index.url = None
    with pytest.raises(ValueError):
        QdrantStore(cfg)


# ── 索引门面 ──────────────────────────────────────────────────────────


def _chunks_for_store():
    parent = Chunk(
        chunk_id="doc1:p000", doc_id="doc1", parent_id=None, is_parent=True,
        text="[t.md > 章]\n父块正文", raw_text="父块正文", title_path=["章"], source="t.md", tokens=10,
    )
    child = Chunk(
        chunk_id="doc1:c0000", doc_id="doc1", parent_id="doc1:p000",
        text="[t.md > 章]\n华东区带宽采购单价 3.2 万元", raw_text="华东区带宽采购单价 3.2 万元",
        title_path=["章"], source="t.md", tokens=12,
    )
    child2 = Chunk(
        chunk_id="doc1:c0001", doc_id="doc1", parent_id="doc1:p000",
        text="[t.md > 章]\n华北区带宽采购单价 2.8 万元", raw_text="华北区带宽采购单价 2.8 万元",
        title_path=["章"], source="t.md", tokens=12,
    )
    return [parent, child, child2]


@pytest.fixture
def store(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_store")
    cfg.index.backend = "memory"
    return IndexStore(cfg)


def test_build_and_stats(store):
    rep = store.build(_chunks_for_store())
    assert rep.chunks_indexed == 2 and rep.parents_written == 1
    assert rep.index_version == store.cfg.index_version
    assert not rep.partial_failure
    s = store.stats()
    assert s["chunks"] == 2 and s["parents"] == 1 and s["documents"] == 1
    assert s["bm25_available"] is True


def test_parent_of_and_get_chunks(store):
    store.build(_chunks_for_store())
    parent = store.parent_of("doc1:c0000")
    assert parent is not None and parent.chunk_id == "doc1:p000"
    assert store.parent_of("doc1:p000") is None
    got = store.get_chunks(["doc1:c0000", "doc1:p000"])
    assert {c.chunk_id for c in got} == {"doc1:c0000", "doc1:p000"}


def test_version_mismatch_rejects_load(store):
    store.build(_chunks_for_store())
    from agentic_rag.index.store import IndexVersionMismatch

    other = IndexStore(store.cfg)
    other.cfg = store.cfg.clone()
    other.cfg.dashscope.embedding_dim = 768  # 改了哈希因子
    other._loaded = False
    with pytest.raises(IndexVersionMismatch):
        other.load()


def test_load_restores_chunks_and_bm25(store):
    store.build(_chunks_for_store())
    fresh = IndexStore(store.cfg)
    fresh.load()
    assert len(fresh.all_chunks()) == 2
    hits = fresh.bm25.search("华东区带宽采购单价", 2)
    assert hits and hits[0][0] == "doc1:c0000"


def test_incremental_upsert_triggers_full_bm25_rebuild(store):
    store.build(_chunks_for_store())
    extra = Chunk(
        chunk_id="doc2:c0000", doc_id="doc2", parent_id=None,
        text="[t2.md]\n设备型号 XR-2400 双电源冗余", raw_text="设备型号 XR-2400 双电源冗余",
        title_path=[], source="t2.md", tokens=10,
    )
    store.upsert_documents([extra])
    assert len(store.all_chunks()) == 3
    hits = store.bm25.search("XR-2400", 3)
    assert hits and hits[0][0] == "doc2:c0000"  # 新文档进倒排 = 全量重建生效


def test_user_dict_missing_is_warned_not_fatal(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_dict")
    cfg.index.backend = "memory"
    cfg.index.user_dict_path = str(tmp_path / "no_such_dict.txt")
    st = IndexStore(cfg)
    rep = st.build(_chunks_for_store())
    assert "user_dict_missing" in rep.warnings
    assert rep.chunks_indexed == 2


# ── 加载期一致性校验（回归：只比 version 不够）────────────────────────


def _built_store(cfg, tmp_path, name="idx_sig"):
    cfg.index.path = str(tmp_path / name)
    cfg.index.backend = "memory"
    st = IndexStore(cfg)
    st.build([_mk(cid, t) for cid, t in DOCS])
    return st


def test_load_rejects_bm25_signature_mismatch(cfg, tmp_path):
    """改 BM25 打分参数：index_version 不变（只由模型+切分算出），必须靠 BM25 签名拦住。"""
    _built_store(cfg, tmp_path)
    IndexStore(cfg).load()  # 同配置下应当正常加载
    old_sig = cfg.bm25_signature

    cfg.index.bm25_k1 = 1.2  # 打分参数变了 → 已落盘的 BM25 与当前配置不是一回事
    assert cfg.bm25_signature != old_sig
    fresh = IndexStore(cfg)
    with pytest.raises(IndexVersionMismatch) as ei:
        fresh.load()
    assert "BM25 参数/用户词典不一致" in str(ei.value)
    assert fresh.cfg.index_version == cfg.index_version  # 版本确实没变，只能靠签名拦


def test_query_time_switches_do_not_require_rebuild(cfg, tmp_path):
    """查询期开关（use_bm25 / min_candidates …）改的是怎么用索引，不是索引里有什么。

    把它们算进加载校验会误伤消融变体：A1 只关 BM25 路，本不该要求重建索引。
    """
    _built_store(cfg, tmp_path)
    before = cfg.retrieval_signature
    cfg.retrieval.use_bm25 = False
    cfg.retrieval.min_candidates = 2
    assert cfg.retrieval_signature != before, "整体签名确实变了"
    assert cfg.bm25_signature == _built_sig(cfg), "但索引期签名没变"

    fresh = IndexStore(cfg)
    fresh.load()  # 不该抛 IndexVersionMismatch
    assert not any("signature" in w for w in fresh.warnings)


def _built_sig(cfg) -> str:
    from agentic_rag.config import bm25_signature

    return bm25_signature(cfg)


def test_load_warns_when_index_built_degraded(cfg, tmp_path):
    """向量路失败过的索引必须自带 degraded 标记，并在加载时说出来。"""
    st = _built_store(cfg, tmp_path)
    meta = st.vector_store.load_meta()
    assert meta["degraded"] is False
    meta["degraded"] = True
    st.vector_store.save_meta(meta)

    fresh = IndexStore(cfg)
    fresh.load()
    assert any("index_degraded" in w for w in fresh.warnings)
    assert fresh.stats()["warnings"] == fresh.warnings


def test_load_warns_on_user_dict_mismatch(cfg, tmp_path):
    st = _built_store(cfg, tmp_path)
    idx = st.bm25
    idx.user_dict_sha = "deadbeefdeadbeef"
    idx.save(st.bm25_path)
    fresh = IndexStore(cfg)
    fresh.load()
    assert any("user_dict_mismatch" in w for w in fresh.warnings)


def test_load_user_dict_is_cwd_independent(cfg, monkeypatch, tmp_path):
    """路径相对项目根解析：换个 CWD 跑，词典不能静默消失。"""
    from agentic_rag.index.bm25 import load_user_dict

    words = load_user_dict("data/user_dict.txt")
    assert words, "项目根的词典应当能加载"
    monkeypatch.chdir(tmp_path)
    assert load_user_dict("data/user_dict.txt") == words


def test_bm25_build_owns_the_user_dict_fingerprint(cfg, tmp_path):
    """词典指纹必须由 `BM25Index.build()` 自己写，不能靠调用方事后补。

    早先 build() 里判 `isinstance(user_dict, str)`，而调用方传的是 `list[str]`
    → 分支恒 False，这行成了死代码；真正的值反而是 store 在 build() 之外另行赋值。
    于是多了一个"谁补了才算数"的隐式前提：`build_from_chunks()` 那条路径没人补，
    指纹永远是 "none"，而 `_check_user_dict_sha` 拿 "none" 和真实指纹一比，
    会报一个"词典漂移"的假警告。写入方必须只有一个。
    """
    from agentic_rag.index.bm25 import BM25Index, build_from_chunks, load_user_dict

    d = tmp_path / "dict.txt"
    d.write_text("带宽 100 n\n吉比特 100 n\n", encoding="utf-8")
    words = load_user_dict(str(d))
    assert words, "自定义词典应当能加载"

    idx = BM25Index()
    idx.build([_mk(cid, t) for cid, t in DOCS], user_dict=words, user_dict_path=str(d))
    assert idx.user_dict_sha != "none", "build() 必须自己落定指纹"

    class _Idx:
        bm25_k1, bm25_b, user_dict_path = 1.5, 0.75, str(d)

    got = build_from_chunks([_mk(cid, t) for cid, t in DOCS], _Idx())
    assert got.user_dict_sha == idx.user_dict_sha, "两条构造路径的指纹必须一致"


def test_user_dict_fingerprint_is_none_when_dict_missing(cfg, tmp_path):
    """词典没加载成功时指纹必须是 "none"，与这一次实际的分词口径对齐。"""
    idx = BM25Index()
    idx.build(
        [_mk(cid, t) for cid, t in DOCS],
        user_dict=None,
        user_dict_path=str(tmp_path / "no_such_dict.txt"),
    )
    assert idx.user_dict_sha == "none"


def test_built_index_carries_user_dict_sha(cfg, tmp_path):
    """经 IndexStore 构建的索引，BM25 必须带上指纹（加载期的漂移校验才有效）。"""
    st = _built_store(cfg, tmp_path)
    assert st.bm25.user_dict_sha != "none"


def test_upsert_is_idempotent_and_updates_meta(cfg, tmp_path):
    """重复摄取同一文档不能堆重复行，meta 计数也必须跟着走。

    物化视图是追加写的：不去重的话同一个块会被 BM25 索引两次（IDF 与 docs 计数全错），
    而 meta 里的数字还停在构建时——两者不一致且一声不吭。
    """
    chunks = [_mk(cid, t) for cid, t in DOCS]
    st = _built_store(cfg, tmp_path)
    before = len(st.all_chunks())

    st.upsert_documents(chunks)  # 完全相同的一批 → 应当什么都不加
    st2 = IndexStore(cfg)
    st2.load()
    assert len(st2.all_chunks()) == before
    assert st2.stats()["chunks"] == before

    st.upsert_documents([_mk("d9", "新增 文档 内容")])
    st3 = IndexStore(cfg)
    st3.load()
    assert len(st3.all_chunks()) == before + 1
    assert st3.stats()["chunks"] == before + 1  # meta 计数跟着长


def test_load_rejects_index_built_by_other_provider_mode(cfg, tmp_path):
    """Mock 与真实 API 用同一个模型名和维度 → index_version 一模一样。

    不单记 provider_mode，切回真实模式就会静默加载一批假向量：版本校验通过、
    余弦分照样算得出来，只是毫无意义。这是"两个模式都能跑、但只有一个是对的"。
    """
    st = _built_store(cfg, tmp_path)
    assert cfg.index_version == IndexStore(cfg).cfg.index_version
    meta = st.vector_store.load_meta()
    assert meta["provider_mode"] == cfg.provider_mode

    meta["provider_mode"] = "mock" if cfg.provider_mode != "mock" else "dashscope"
    st.vector_store.save_meta(meta)
    fresh = IndexStore(cfg)
    with pytest.raises(IndexVersionMismatch) as ei:
        fresh.load()
    assert "向量供给方不一致" in str(ei.value)


def test_load_warns_when_provider_mode_unrecorded(cfg, tmp_path):
    """旧索引没写这一项 → 只警告不拦（向前兼容），但要说出来。"""
    st = _built_store(cfg, tmp_path)
    meta = st.vector_store.load_meta()
    meta.pop("provider_mode")
    st.vector_store.save_meta(meta)
    fresh = IndexStore(cfg)
    fresh.load()
    assert any("provider_mode" in w for w in fresh.warnings)
