"""集中配置 + 启动期一致性校验（03 §1）。

两条硬规矩：
1. **只有四项从环境读取**（凭据与端点），其余参数一律走配置树默认值或 CLI 显式参数。
   理由是可复现性：`retrieval_signature` 与 `EvalReport` 记录的是配置树的序列化结果，
   若任何参数都能从环境变量悄悄注入，同一份报告就无法复现。
2. **模块内不得出现数字字面量**，阈值全部在这里声明。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .prompts.loader import all_prompts
from .utils import sha256_file

# 项目根：src/agentic_rag/config.py -> 上两级。相对路径一律以它为基准，
# 这样 CLI 在任何工作目录下运行都指向同一份 data/。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ─────────────────────────────── 配置树 ───────────────────────────────


@dataclass
class DashScopeConfig:
    api_key: str | None = None  # None / 空 → 自动启用 Mock 全家桶
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    native_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    llm_main: str = "qwen-plus"
    llm_light: str = "qwen-flash"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    rerank_model: str = "qwen3-rerank"
    timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff: tuple[float, ...] = (0.6, 1.2, 2.4)


@dataclass
class QualityConfig:
    scanned_chars_per_page: int = 50
    digit_loss_warn: float = 0.10
    digit_loss_reject: float = 0.30
    digit_count_min_for_check: int = 20
    raw_char_min_for_check: int = 500
    ocr_min_chars: int = 200
    text_coverage_warn: float = 0.60
    text_coverage_reject: float = 0.40
    table_recall_warn: float = 0.80
    heading_jump_warn: int = 3
    on_reject: str = "quarantine"  # quarantine | skip


@dataclass
class IngestionConfig:
    chunk_strategy: str = "parent_child"  # parent_child | structural
    parent_chunk_tokens: int = 1024
    child_chunk_tokens: int = 256
    child_overlap_tokens: int = 48
    atomic_max_tokens: int = 2048
    table_split_threshold_tokens: int = 1024
    use_docling: bool = False
    docling_table_density_threshold: float = 0.15
    workers: int = 4
    run_probe_after_build: bool = True
    # PDF 后端：pymupdf（AGPL-3.0，默认，快）| pypdf（BSD，用于规避 AGPL）。
    # 探测（quality.probe_before_parse）与 raw 基线（quality.raw_baseline）
    # 必须与 PdfParser 共用同一个后端工厂，否则许可规避只生效一半（J10）。
    pdf_backend: str = "pymupdf"


@dataclass
class IndexConfig:
    backend: str = "qdrant"  # qdrant | qdrant_server | memory
    path: str = "data/index"
    url: str | None = None
    collection: str = "agentic_rag"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    user_dict_path: str = "data/user_dict.txt"


@dataclass
class RetrievalConfig:
    vector_top_k: int = 50
    bm25_top_k: int = 50
    fusion_top_k: int = 50
    fusion_strategy: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    weighted_weights: tuple[float, float] = (0.5, 0.5)
    use_vector: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    parent_backfill: bool = True  # B3 消融：False 时上下文直接用子块，不回填父块
    rerank_top_k: int = 5
    rerank_max_tokens: int = 4000
    table_boost: float = 1.2
    filter_tables_only: bool = False
    min_candidates: int = 3
    # 设计初值。标定脚本 scripts/calibrate_k1.py 在本语料 33,477 条真实
    # text-embedding-v4 向量上测得参考区间 0.47–0.54，即 0.35 偏松。
    # 默认值保留 0.35 是为了维持与既有结果的可比性：阈值同时影响 K1 判定
    # 与反思触发频率，换语料或重跑评测前应重新标定。
    sufficiency_vector_floor: float = 0.35
    unanswerable_rerank_floor: float = 0.35


@dataclass
class AgentConfig:
    intents: tuple[str, ...] = ("simple", "multi_hop", "table_numeric", "chitchat")
    skip_rewrite_for_simple: bool = True
    allow_reflect_for_simple: bool = False
    max_sub_queries: int = 4
    history_max_turns: int = 3
    max_retrieve_iterations: int = 3  # L2：三级策略升级需要两次重检
    max_generate_iterations: int = 2


@dataclass
class GenerationConfig:
    temperature: float = 0.1
    max_tokens: int = 2048
    lang: str = "zh"
    max_citations: int = 5
    faithfulness_threshold: float = 0.80
    regenerate_on_unsupported: bool = False
    max_context_chunks: int = 5


@dataclass
class CacheConfig:
    enabled: bool = True
    include_config_signature: bool = True
    dir: str = "data/cache"
    ttl_s: int = 604800
    max_entries: int = 100_000
    # embedding / rerank 级的模块（cache/compute_cache.py）已实现但未接入调用链，
    # 接线前默认只启用已生效的 result 级（见 docs/02 §5.4 接线状态）。
    enabled_levels: tuple[str, ...] = ("result",)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent_queries: int = 4
    queue_size: int = 256


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    trace_dir: str = "data/traces"
    trace_keep_top_k: int = 50  # 必须 ≥ 评测要上报的最大 K，否则事后算不出


@dataclass
class Config:
    profile: str = "dev"  # dev | eval
    dashscope: DashScopeConfig = field(default_factory=DashScopeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    @property
    def index_version(self) -> str:
        return compute_index_version(self)

    @property
    def bm25_signature(self) -> str:
        return bm25_signature(self)

    @property
    def retrieval_signature(self) -> str:
        return retrieval_signature(self)

    @property
    def generation_signature(self) -> str:
        return generation_signature(self)

    @property
    def provider_mode(self) -> str:
        return "dashscope" if (self.dashscope.api_key or "").strip() else "mock"

    @property
    def index_dir(self) -> Path:
        return project_path(self.index.path)

    @property
    def cache_dir(self) -> Path:
        return project_path(self.cache.dir)

    @property
    def trace_dir(self) -> Path:
        return project_path(self.observability.trace_dir)

    def clone(self, **kw) -> "Config":
        """浅拷贝并覆盖若干子配置（消融实验改配置用）。

        必须重跑校验：`replace` 绕过了 `__post_init__`/`get_config(validate=True)`
        那一道，而校验里守着 `trace_keep_top_k ≥ fusion_top_k`、
        `use_vector 或 use_bm25 至少一个为真` 这类不变量。消融改的就是这些开关，
        改坏了不报错的话，跑出来的变体指标是"配置已自相矛盾"的产物，
        而报告上不会有任何提示。
        """
        new = replace(self, **kw)
        errs = validate_config(new)
        if errs:
            raise ValueError(f"配置校验未通过（clone 后的变体配置）：{'; '.join(errs)}")
        return new


# ─────────────────────────────── 环境读取 ───────────────────────────────


def _load_dotenv() -> dict[str, str]:
    """读项目根的 `.env`（不引入 python-dotenv 依赖）。已存在的环境变量优先。"""
    env: dict[str, str] = {}
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def get_config(env: dict[str, str] | None = None, validate: bool = True) -> Config:
    """构造配置：`.env` < 进程环境变量 < 显式传入的 env。"""
    merged = _load_dotenv()
    merged.update({k: v for k, v in os.environ.items()})
    if env:
        merged.update(env)

    cfg = Config(profile=merged.get("AR_PROFILE", "dev"))
    ds = cfg.dashscope
    ds.api_key = merged.get("DASHSCOPE_API_KEY") or None
    ds.base_url = merged.get("DASHSCOPE_BASE_URL", ds.base_url)
    ds.native_base_url = merged.get("DASHSCOPE_NATIVE_BASE_URL", ds.native_base_url)

    if validate:
        validate_config(cfg)
    return cfg


# ─────────────────────────────── 指纹 ───────────────────────────────


def _chunk_factors(cfg: Config) -> str:
    """一切决定 chunk 内容与 chunk_id 的切分参数（H19）。"""
    ing = cfg.ingestion
    return ":".join(
        [
            ing.chunk_strategy,
            str(ing.parent_chunk_tokens),
            str(ing.child_chunk_tokens),
            str(ing.child_overlap_tokens),
            str(ing.atomic_max_tokens),
            str(ing.table_split_threshold_tokens),
        ]
    )


def compute_index_version(cfg: Config) -> str:
    ds = cfg.dashscope
    key = f"{ds.embedding_model}:{ds.embedding_dim}:{_chunk_factors(cfg)}"
    return "v" + hashlib.sha1(key.encode()).hexdigest()[:8]


def user_dict_sha256(path: str) -> str:
    """词典不存在时返回 'none'，不抛异常——允许无词典运行，但指纹必须与有词典时不同。"""
    return sha256_file(project_path(path))[:16] if path else "none"


def bm25_signature(cfg: Config) -> str:
    """**索引期**参数指纹：只含"会影响已落盘索引内容"的那几项。

    必须和 `retrieval_signature` 分开，因为后者还含 `use_bm25` / `use_vector` /
    `fusion_strategy` / `min_candidates` 这类**查询期开关**——它们改的是怎么用索引，
    不是索引里有什么。混在一个签名里做加载校验，A1（关 BM25 路）这种纯查询期变体
    会被误判成"索引需要重建"，消融根本跑不起来。
    """
    factors = [
        cfg.index.bm25_k1,
        cfg.index.bm25_b,
        user_dict_sha256(cfg.index.user_dict_path),
    ]
    return hashlib.sha1("|".join(map(str, factors)).encode()).hexdigest()[:12]


def retrieval_signature(cfg: Config) -> str:
    """**整体**检索指纹（索引期 + 查询期），用于结果缓存键与 EvalReport 存证。"""
    r, ing = cfg.retrieval, cfg.ingestion
    factors = [
        r.use_vector,
        r.use_bm25,
        r.use_rerank,
        r.parent_backfill,
        r.fusion_strategy,
        r.rrf_k,
        r.weighted_weights,
        r.vector_top_k,
        r.bm25_top_k,
        r.fusion_top_k,
        # K1：这两个决定是否触发反思，从而改变最终答案
        r.min_candidates,
        r.sufficiency_vector_floor,
        r.table_boost,
        r.filter_tables_only,
        # M15：`rerank_top_k` 决定回填与顺位补齐的池子深度（改了就换一批上下文块）；
        # 编排侧的四个旋钮决定"检索跑几轮、拆几条子查询、simple 进不进反思"——
        # 它们直接改变最终召回，却既不在 index_version 里也不在别处，
        # 漏掉的话 UI 现场切这些开关会命中旧答案（与 J6 同一族故障）。
        r.rerank_top_k,
        cfg.agent.allow_reflect_for_simple,
        cfg.agent.skip_rewrite_for_simple,
        cfg.agent.max_retrieve_iterations,
        cfg.agent.max_sub_queries,
        cfg.dashscope.embedding_model,
        cfg.dashscope.embedding_dim,
        ing.chunk_strategy,
        ing.parent_chunk_tokens,
        ing.child_chunk_tokens,
        ing.child_overlap_tokens,
        ing.atomic_max_tokens,
        ing.table_split_threshold_tokens,
        # L6：BM25 打分参数与词典指纹
        cfg.index.bm25_k1,
        cfg.index.bm25_b,
        user_dict_sha256(cfg.index.user_dict_path),
    ]
    return hashlib.sha1("|".join(map(str, factors)).encode()).hexdigest()[:12]


def generation_signature(cfg: Config) -> str:
    g, ds = cfg.generation, cfg.dashscope
    factors = [
        ds.llm_main,
        ds.llm_light,
        g.temperature,
        g.max_tokens,
        g.lang,
        g.max_citations,
        g.max_context_chunks,
        # L6：这两个决定是否重生成，从而改变最终答案
        g.faithfulness_threshold,
        g.regenerate_on_unsupported,
        cfg.agent.max_generate_iterations,  # M15：生成跑几轮同样改变最终答案
        *[p.schema_version for p in all_prompts()],
    ]
    return hashlib.sha1("|".join(map(str, factors)).encode()).hexdigest()[:12]


def history_fingerprint(history: list[dict] | None, max_turns: int) -> str:
    """只取最近 max_turns 轮，且只取 role/content（L3）。

    截断长度必须与 `rewrite_node` 实际能看到的历史长度一致，
    否则同一个语义上下文会因缓存键截断位置不同而分裂成多个键。
    """
    if not history:
        return "-"
    turns = history[-max_turns * 2 :]  # 一轮 = user + assistant
    raw = "\x1f".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in turns)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ─────────────────────────────── 启动校验 ───────────────────────────────


class ConfigError(ValueError):
    """启动期一致性校验失败。"""


_ALIASES = {"qwen-plus", "qwen-flash", "qwen-max", "qwen-turbo"}


def validate_config(cfg: Config) -> None:
    """03 §1 的启动校验表，逐条实现。失败抛 ConfigError。"""
    errs: list[str] = []
    ing, r = cfg.ingestion, cfg.retrieval
    q, a, g = cfg.quality, cfg.agent, cfg.generation

    def chk(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)

    # 切分尺寸
    chk(ing.child_chunk_tokens < ing.parent_chunk_tokens, "child_chunk_tokens 必须 < parent_chunk_tokens")
    chk(ing.parent_chunk_tokens <= ing.atomic_max_tokens, "parent_chunk_tokens 必须 ≤ atomic_max_tokens")
    chk(ing.table_split_threshold_tokens <= ing.atomic_max_tokens, "table_split_threshold_tokens 必须 ≤ atomic_max_tokens")
    chk(
        0 < ing.child_overlap_tokens < ing.child_chunk_tokens,
        "child_overlap_tokens 必须 ∈ (0, child_chunk_tokens)",
    )

    # 召回与融合宽度
    chk(r.rerank_top_k <= r.fusion_top_k, "rerank_top_k 必须 ≤ fusion_top_k")
    chk(r.fusion_top_k <= r.vector_top_k + r.bm25_top_k, "fusion_top_k 必须 ≤ vector_top_k + bm25_top_k")
    chk(r.fusion_top_k >= max(r.vector_top_k, r.bm25_top_k), "fusion_top_k 必须 ≥ max(vector_top_k, bm25_top_k)")
    chk(
        cfg.observability.trace_keep_top_k >= r.fusion_top_k,
        "trace_keep_top_k 必须 ≥ fusion_top_k（埋点窄于召回宽度会让指标事后算不出）",
    )
    chk(64 <= cfg.dashscope.embedding_dim <= 2048, "embedding_dim 必须 ∈ [64, 2048]")

    # 质量阈值单调性
    chk(0 < q.digit_loss_warn < q.digit_loss_reject < 1, "digit_loss: 需 0 < warn < reject < 1")
    chk(0 < q.text_coverage_reject < q.text_coverage_warn < 1, "text_coverage: 需 0 < reject < warn < 1")
    chk(0 < q.table_recall_warn < 1, "table_recall_warn 必须 ∈ (0,1)")
    chk(q.raw_char_min_for_check > 0, "raw_char_min_for_check 必须 > 0")
    chk(q.ocr_min_chars > 0, "ocr_min_chars 必须 > 0")
    chk(q.on_reject in ("quarantine", "skip"), "on_reject 必须 ∈ {quarantine, skip}")

    # 检索判据
    chk(0 < r.unanswerable_rerank_floor < 1, "unanswerable_rerank_floor 必须 ∈ (0,1)")
    chk(-1 < r.sufficiency_vector_floor < 1, "sufficiency_vector_floor 必须 ∈ (-1,1)")
    chk(r.min_candidates >= 1, "min_candidates 必须 ≥ 1")
    chk(r.use_vector or r.use_bm25, "use_vector 与 use_bm25 不得同时为 False")

    # 生成与编排
    chk(g.max_context_chunks <= r.rerank_top_k, "max_context_chunks 必须 ≤ rerank_top_k")
    chk(a.max_retrieve_iterations >= 1, "max_retrieve_iterations 必须 ≥ 1")
    chk(a.max_retrieve_iterations <= 3, "max_retrieve_iterations 必须 ≤ 3（策略升级共三级，再往上收益为负）")
    chk(a.max_generate_iterations >= 1, "max_generate_iterations 必须 ≥ 1")
    chk(1 <= a.max_sub_queries <= 8, "max_sub_queries 必须 ∈ [1, 8]")
    chk(a.history_max_turns >= 0, "history_max_turns 必须 ≥ 0")
    chk(g.lang in ("zh", "en"), "generation.lang 必须 ∈ {zh, en}")
    chk(1 <= g.max_citations <= g.max_context_chunks, "max_citations 必须 ∈ [1, max_context_chunks]")
    chk(0 <= g.faithfulness_threshold <= 1, "faithfulness_threshold 必须 ∈ [0,1]")

    # 枚举白名单
    chk(ing.chunk_strategy in ("parent_child", "structural"), "chunk_strategy 取值非法")
    chk(r.fusion_strategy in ("rrf", "weighted"), "fusion_strategy 取值非法")
    chk(cfg.index.backend in ("qdrant", "qdrant_server", "memory"), "index.backend 取值非法")
    chk(ing.pdf_backend in ("pymupdf", "pypdf"), "ingestion.pdf_backend 取值非法")
    if cfg.index.backend == "qdrant_server":
        chk(bool(cfg.index.url), "backend=qdrant_server 时 url 必填（漏了会静默退回 local 模式）")
    chk(cfg.server.max_concurrent_queries >= 1, "max_concurrent_queries 必须 ≥ 1")
    chk(cfg.profile in ("dev", "eval"), "profile 必须 ∈ {dev, eval}")

    # eval 档的两条硬校验（来自 2026-10-21 实测：别名会静默漂移）
    if cfg.profile == "eval":
        if cfg.dashscope.llm_main in _ALIASES or cfg.dashscope.llm_light in _ALIASES:
            errs.append("AR_PROFILE=eval 时 llm_main / llm_light 不得为别名，必须锁版本号")
        if cfg.cache.enabled:
            errs.append("AR_PROFILE=eval 时 cache.enabled 必须为 False")

    if errs:
        raise ConfigError("配置校验失败：\n  - " + "\n  - ".join(errs))
