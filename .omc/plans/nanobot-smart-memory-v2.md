# Nanobot 智能记忆系统 v2.0

> 状态: **Final Plan**
> 创建时间: 2026-02-11
> 技术栈: 纯 Python (sentence-transformers + ChromaDB)

> 补充说明（实现向）：本文件在 v2.0 方案基础上补齐可落地细节（数据模型、索引生命周期、降级策略、配置对齐等），并修正示例代码中「检索返回值」与「分层逻辑」不一致的问题。

---

## 一、方案概述

### 1.1 与 v1.0 的变化

| 维度 | v1.0 (QMD) | v2.0 (纯 Python) |
|------|------------|------------------|
| Embedding | QMD 本地模型 | sentence-transformers |
| 向量存储 | QMD sqlite-vec | ChromaDB |
| 运行时 | Bun + Node.js | **纯 Python** |
| 模型大小 | ~2GB | **~118MB** |
| 安装 | bun install | **pip install** |

### 1.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                Nanobot Smart Memory v2.0                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  写入层              存储层              检索层                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ AgentHooks  │    │ Markdown    │    │ Hybrid Search       │ │
│  │ ├─ToolObs   │───▶│ Files       │◀───│ ├─ BM25 (FTS5)      │ │
│  │ └─Summary   │    │ (不变)      │    │ └─ Vector (Chroma)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                            │                    ▲               │
│                            ▼                    │               │
│                     ┌─────────────┐    ┌─────────────────────┐ │
│                     │ .storage/   │    │ sentence-transformers│ │
│                     │ ├─ chroma/  │    │ multilingual-MiniLM │ │
│                     │ └─ fts.db   │    │ (118MB 本地)        │ │
│                     └─────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、技术选型

### 2.1 Embedding 模型

**选择：`paraphrase-multilingual-MiniLM-L12-v2`**

| 属性 | 值 |
|------|----|
| 大小 | 118MB |
| 维度 | 384 |
| 语言 | 50+ (含中英文) |
| 下载量 | 1800万+ |

### 2.2 依赖

```bash
pip install sentence-transformers chromadb
```

---

## 三、文件结构

### 新增文件

```
nanobot/
├── agent/
│   ├── hooks.py           # Agent 生命周期 Hooks
│   ├── observation.py     # 工具观察记录
│   ├── summarizer.py      # 会话摘要生成
│   ├── memory.py          # HybridMemoryStore (重构)
│   └── retrieval.py       # 混合检索
├── storage/
│   ├── embedding.py       # Embedding 服务
│   ├── vectorstore.py     # ChromaDB 封装
│   └── fts.py             # FTS5 全文搜索
```

### 数据存储

```
workspace/
├── memory/                 # Markdown (不变)
│   ├── MEMORY.md
│   └── 2026-02-11.md
├── .storage/               # 新增：索引
│   ├── chroma/
│   ├── fts.sqlite3
│   └── manifest.jsonl       # 索引清单（幂等/增量/删除）
```

---

## 四、核心模块

### 4.1 Embedding 服务

```python
# nanobot/storage/embedding.py

from sentence_transformers import SentenceTransformer

class LocalEmbedding:
    MODELS = {
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 118MB
        "chinese": "BAAI/bge-small-zh-v1.5",  # 95MB
        "english": "all-MiniLM-L6-v2",  # 22MB
    }

    def __init__(self, model_name: str = "multilingual"):
        actual_model = self.MODELS.get(model_name, model_name)
        self.model = SentenceTransformer(actual_model)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
```

### 4.2 向量存储 (ChromaDB)

```python
# nanobot/storage/vectorstore.py

import chromadb
from dataclasses import dataclass

@dataclass
class VectorSearchResult:
    id: str
    text: str
    score: float
    metadata: dict

class VectorStore:
    def __init__(self, storage_path: Path, embedding: LocalEmbedding):
        self.embedding = embedding
        self.client = chromadb.PersistentClient(path=str(storage_path))
        self.collection = self.client.get_or_create_collection("memory")

    def add(self, text: str, metadata: dict = None) -> str:
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        embedding = self.embedding.embed(text)
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
        return doc_id

    def search(self, query: str, limit: int = 5) -> list[VectorSearchResult]:
        query_embedding = self.embedding.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        # 转换距离为相似度
        return [
            VectorSearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                score=1 / (1 + results["distances"][0][i]),
                metadata=results["metadatas"][0][i]
            )
            for i in range(len(results["ids"][0]))
        ]
```

### 4.3 全文搜索 (FTS5)

```python
# nanobot/storage/fts.py

import sqlite3

class FullTextSearch:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(id, text, title, source)
        """)

    def add(self, text: str, doc_id: str, title: str = ""):
        self.conn.execute(
            "INSERT INTO docs_fts VALUES (?, ?, ?, ?)",
            (doc_id, text, title, "")
        )
        self.conn.commit()

    def search(self, query: str, limit: int = 10):
        cursor = self.conn.execute("""
            SELECT
                id,
                text,
                bm25(docs_fts) as bm25_score,
                snippet(docs_fts, 1, '[', ']', '...', 12) as snippet
            FROM docs_fts
            WHERE docs_fts MATCH ?
            ORDER BY bm25_score LIMIT ?
        """, (query, limit))
        return cursor.fetchall()
```

### 4.4 混合检索（修订：统一返回结构 + 按 rank 分层）

```python
# nanobot/agent/retrieval.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RetrievedChunk:
    """
    混合检索的统一返回结构，用于后续的分层、裁剪与上下文格式化。

    说明：
    - RRF 的分数是 rank 融合分数，量级通常远小于 0.5，因此不建议用 0.5/0.3
      这种阈值做分层；分层更推荐按 rank/top-k 来做。
    """

    id: str
    text: str
    fused_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    snippet: str | None = None
    is_full: bool = False


class HybridRetrieval:
    def __init__(self, workspace: Path, embedding_model: str = "multilingual"):
        self.embedding = LocalEmbedding(embedding_model)
        self.vector_store = VectorStore(workspace / ".storage/chroma", self.embedding)
        self.fts = FullTextSearch(workspace / ".storage/fts.sqlite3")

    def index_document(self, text: str, title: str = "", metadata: dict[str, Any] | None = None) -> str:
        """
        将文档同时写入 VectorStore 与 FTS。

        约定：返回的 doc_id 必须在两套索引中一致，便于 RRF 融合与删除/更新。
        """
        doc_id = self.vector_store.add(text=text, metadata={"title": title, **(metadata or {})})
        self.fts.add(text=text, doc_id=doc_id, title=title)
        return doc_id

    def search(self, query: str, limit: int = 10, prefetch_multiplier: int = 2) -> list[RetrievedChunk]:
        # 1) 向量搜索（更擅长语义）
        vector_results = self.vector_store.search(query, limit * prefetch_multiplier)

        # 2) 全文搜索（更擅长关键词；顺便拿 snippet 充当“摘要”）
        # fts_results: list[tuple[id, text, bm25_score, snippet]]
        fts_results = self.fts.search(query, limit * prefetch_multiplier)

        # 3) RRF 融合（只用 rank，不依赖 bm25 数值量级）
        fused_scores = self._rrf_scores(vector_results, fts_results, k=60)

        by_id: dict[str, RetrievedChunk] = {}

        # Prefer vector store text as "full text" source
        for r in vector_results:
            by_id[r.id] = RetrievedChunk(
                id=r.id,
                text=r.text,
                fused_score=fused_scores.get(r.id, 0.0),
                metadata=r.metadata,
            )

        # Attach snippet from FTS (cheap "摘要") and fallback to FTS text if missing
        for doc_id, text, _bm25_score, snippet in fts_results:
            chunk = by_id.get(doc_id)
            if not chunk:
                chunk = RetrievedChunk(id=doc_id, text=text, fused_score=fused_scores.get(doc_id, 0.0))
                by_id[doc_id] = chunk
            if snippet:
                chunk.snippet = snippet

        ordered = sorted(by_id.values(), key=lambda c: c.fused_score, reverse=True)
        return ordered[:limit]

    def _rrf_scores(self, vec_results, fts_results, k: int = 60) -> dict[str, float]:
        """Reciprocal Rank Fusion（只用 rank 融合，避免不同 score 量级不一致）"""
        scores: dict[str, float] = {}
        for rank, r in enumerate(vec_results, 1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank)
        for rank, r in enumerate(fts_results, 1):
            doc_id = r[0]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return scores

    def progressive_retrieve(
        self,
        query: str,
        max_full_docs: int = 3,
        max_snippets: int = 5,
    ) -> list[RetrievedChunk]:
        """两段式渐进检索：top-k 全文 + 后续 snippet/截断文本"""
        results = self.search(query, limit=max_full_docs + max_snippets)
        for r in results[:max_full_docs]:
            r.is_full = True
        return results
```

### 4.5 Agent Hooks

```python
# nanobot/agent/hooks.py

from abc import ABC
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ToolExecution:
    tool_name: str
    arguments: dict
    result: str
    error: Exception | None
    timestamp: datetime
    duration_ms: float
    session_key: str

class AgentHook(ABC):
    async def on_message_received(self, session_key, content): pass
    async def on_tool_executed(self, execution: ToolExecution): pass
    async def on_session_end(self, session_key, history, response): pass

class HookManager:
    def __init__(self):
        self._hooks = []

    def register(self, hook: AgentHook):
        self._hooks.append(hook)

    async def emit_tool_executed(self, execution):
        for hook in self._hooks:
            await hook.on_tool_executed(execution)
```

### 4.6 观察记录 Hook

```python
# nanobot/agent/observation.py

SIGNIFICANT_TOOLS = {"write_file", "edit_file", "exec"}

class ObservationHook(AgentHook):
    def __init__(self, memory_dir: Path, retrieval: HybridRetrieval):
        self.memory_dir = memory_dir
        self.retrieval = retrieval

    async def on_tool_executed(self, execution: ToolExecution):
        if execution.tool_name not in SIGNIFICANT_TOOLS:
            return
        if execution.error:
            return

        # 生成观察内容
        content = self._format(execution)

        # 1. 写入 Markdown
        self._write_markdown(content)

        # 2. 索引到向量库
        self.retrieval.index_document(content, title=execution.tool_name)

    def _format(self, exec):
        return f"""## 🔧 {exec.tool_name} @ {exec.timestamp:%H:%M}

**\1**: {exec.arguments}
**\1**: {exec.result[:300]}...
"""
```

### 4.7 会话摘要 Hook

```python
# nanobot/agent/summarizer.py

SUMMARY_PROMPT = """Summarize this conversation:
1. What user wanted
2. Key actions taken
3. Outcome
Keep under 200 words.

{conversation}
"""

class SummaryHook(AgentHook):
    def __init__(self, memory_dir, retrieval, provider, min_turns=3):
        self.memory_dir = memory_dir
        self.retrieval = retrieval
        self.provider = provider
        self.min_turns = min_turns

    async def on_session_end(self, session_key, history, response):
        if len([m for m in history if m["role"]=="user"]) < self.min_turns:
            return

        # 生成摘要
        summary = await self._generate_summary(history)

        # 写入 MEMORY.md
        self._save(session_key, summary)

        # 索引
        self.retrieval.index_document(summary, title=f"Session: {session_key}")

    async def _generate_summary(self, history):
        conv = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
        resp = await self.provider.chat([{"role": "user", "content": SUMMARY_PROMPT.format(conversation=conv)}])
        return resp.content
```

### 4.8 混合记忆存储

```python
# nanobot/agent/memory.py

class HybridMemoryStore:
    def __init__(self, workspace: Path, embedding_model="multilingual"):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self._retrieval = None
        self._model = embedding_model

    def initialize(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._retrieval = HybridRetrieval(self.workspace, self._model)
        self._index_existing_files()

    @property
    def retrieval(self):
        if not self._retrieval:
            self.initialize()
        return self._retrieval

    async def get_context(self, query: str = None) -> str:
        """获取记忆上下文"""
        parts = []

        # 语义检索
        if query:
            results = self.retrieval.progressive_retrieve(query)
            if results:
                parts.append(self.retrieval.format_context(results))

        # 长期记忆头部
        if (self.memory_dir / "MEMORY.md").exists():
            content = (self.memory_dir / "MEMORY.md").read_text()[-1000:]
            parts.append(f"## Long-term Memory\n{content}")

        return "\n\n---\n\n".join(parts)
```

---

## 五、AgentLoop 集成

```python
# nanobot/agent/loop.py (修改)

class AgentLoop:
    def __init__(self, ..., enable_observations=True, enable_summaries=True):
        # 初始化记忆
        self.memory = HybridMemoryStore(workspace)

        # 初始化 Hooks
        self.hooks = HookManager()

        if enable_observations:
            self.hooks.register(ObservationHook(
                workspace / "memory",
                self.memory.retrieval
            ))

        if enable_summaries:
            self.hooks.register(SummaryHook(
                workspace / "memory",
                self.memory.retrieval,
                provider
            ))

    async def run(self):
        self.memory.initialize()  # 加载 embedding 模型
        # ...

    async def _process_message(self, msg):
        # 使用语义检索构建上下文
        memory_context = await self.memory.get_context(query=msg.content)

        # ... 现有逻辑 ...

        for tool_call in response.tool_calls:
            start = time.time()
            result = await self.tools.execute(tool_call.name, tool_call.arguments)
            duration = (time.time() - start) * 1000

            # 触发 Hook
            await self.hooks.emit_tool_executed(ToolExecution(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=result,
                error=None,
                timestamp=datetime.now(),
                duration_ms=duration,
                session_key=msg.session_key
            ))

        # 会话结束
        await self.hooks.emit_session_end(
            session_key=msg.session_key,
            history=session.messages,
            response=final_content
        )
```

---

## 六、配置

> 说明：nanobot 当前使用 `~/.nanobot/config.json`（camelCase）。建议将 memory 配置纳入 `nanobot/config/schema.py`，并通过现有 loader 的 key 转换/迁移逻辑保持兼容。

```json
{
  "memory": {
    "enabled": true,
    "embeddingModel": "multilingual",
    "enableObservations": true,
    "enableSummaries": true,
    "summaryMinTurns": 3,
    "isolateBySession": false,
    "retrieval": {
      "limit": 8,
      "maxFullDocs": 3,
      "maxSnippets": 5,
      "rrfK": 60,
      "prefetchMultiplier": 2
    }
  }
}
```

---

## 七、安装与使用

### 安装

```bash
# 安装 nanobot 时包含 memory 功能
pip install nanobot[memory]

# 或单独安装依赖
pip install sentence-transformers chromadb
```

### 首次运行

首次运行会自动下载 embedding 模型 (~118MB)：

```
~/.cache/huggingface/hub/
└── models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
```

---

## 八、实施计划

| Phase | 内容 | 时间 |
|-------|------|------|
| 1 | 存储层 (embedding, vectorstore, fts) | 1 天 |
| 2 | 检索层 (retrieval, memory) | 1 天 |
| 3 | Hooks (hooks, observation, summarizer) | 1 天 |
| 4 | 集成 (loop 修改, 测试) | 1 天 |

**总计：4 天**

---

## 九、验收标准

- [ ] `pip install nanobot[memory]` 或 `pip install sentence-transformers chromadb` 成功
- [ ] Embedding 模型自动下载并加载
- [ ] 向量搜索返回相关结果
- [ ] 全文搜索正常工作（含 snippet/高亮片段返回）
- [ ] 混合搜索 RRF 融合正确
- [ ] 工具观察自动记录到 Markdown
- [ ] 工具观察自动索引到向量库
- [ ] 会话摘要生成并保存
- [ ] 重启/重复运行不会造成索引膨胀（幂等：相同输入 doc 数不增长）
- [ ] 文件更新/删除可触发对应索引更新/删除（manifest 或等价机制）
- [ ] Embedding/Chroma 不可用时可降级（至少保留 FTS 或纯 Markdown 记忆）
- [ ] Token 消耗降低 50%+（有基线、样本与统计口径）

---

## 十、优势总结

| 维度 | 说明 |
|------|------|
| **纯 Python** | 无需 Bun/Node.js |
| **轻量模型** | 118MB vs 2GB |
| **简单安装** | pip install |
| **离线可用** | 本地 embedding |
| **中文支持** | 多语言模型 |
| **混合检索** | BM25 + 向量 |
| **自动记录** | Hooks 系统 |
| **会话摘要** | LLM 生成 |

---

## 十一、实现补充（建议纳入 v2.0）

### 11.1 修订建议清单（高优先级）

1. **统一检索返回结构**：`search()` 必须返回可用于分层与格式化的对象（而不是仅返回 id 列表），避免实现阶段出现结构不匹配。
2. **渐进式检索改为按 rank 分层**：RRF 分数不适合作为 0.5/0.3 这种阈值分层依据；推荐 top-k 全文 + 后续 snippet 的两段式策略。
3. **补齐索引生命周期**：明确“首次建索引、增量更新、删除清理、幂等重启”的机制（manifest 或 sqlite 表），否则上线后会快速膨胀且难以维护。
4. **配置对齐到现有 config.json**：避免再引入第二套配置入口（如 `config.yaml`），减少用户与维护成本。
5. **提供降级路径**：embedding 依赖/模型下载失败时仍可运行（例如 FTS-only），避免启动即崩。
6. **加入脱敏与降噪**：Observation/Exec 输出可能包含密钥与隐私内容，需在写入与索引前做最小 redaction + 截断。

### 11.2 数据模型（统一索引/检索/过滤）

建议为每个被索引的文本片段（chunk）维护最小字段集合（写入向量库 metadata + FTS 字段或附表）：

- `doc_id`: 稳定唯一 id（建议包含 `source + chunk_index + content_hash`，避免不同来源相同文本互相覆盖）
- `source_type`: `daily_note` | `long_term` | `observation` | `summary`
- `source_path`: 原始文件路径（若来自文件）
- `title`: 展示用标题（工具名/会话名/文件名）
- `session_key`: 可选；用于会话隔离或过滤
- `timestamp`: 记录时间（用于排序、衰减）
- `content_hash`: 用于幂等与增量更新
- `chunk_index`: 分块序号（长文分块）

理由：没有统一 metadata，会导致“融合/过滤/删除”在实现时变成临时补丁，后期返工代价高。

### 11.3 索引生命周期（幂等 + 增量更新 + 删除）

建议引入 manifest（示例：`workspace/.storage/manifest.jsonl` 或 `workspace/.storage/meta.sqlite3`），至少记录：

- `source_path`、`mtime`、`file_hash`
- 该文件对应的 `doc_id[]`（分块后多个 id）

索引策略：

- 启动/定时：扫描 `workspace/memory/`，若 `mtime/hash` 未变化则跳过；变化则删除旧 doc_id 并重建。
- 删除文件：根据 manifest 找到旧 doc_id 并从 Chroma + FTS 同步删除。
- 观察/摘要写入：写入 Markdown 后，直接对新增 chunk 做 upsert（避免全量重建）。

理由：没有生命周期管理会导致重复索引、召回噪音升高、向量库/FTS 体积膨胀、检索变慢。

### 11.4 检索策略与上下文预算（更可控）

- **融合阶段只使用 rank**：FTS 的 `bm25()` 量级/正负/方向不一定与向量相似度一致；融合用 RRF rank 更稳。
- **中等相关用 snippet**：建议 FTS 返回 `snippet()` 片段作为低成本“摘要”；必要时再引入 LLM 生成摘要（成本更高）。
- **上下文预算**：构建 prompt 时按预算裁剪（Pinned long-term -> Relevant recalls -> Recent summary），避免把 `MEMORY.md` 尾部硬塞固定字数造成噪音。

### 11.5 依赖与降级策略（避免“安装即爆炸”）

- 建议把依赖作为可选 extra：`nanobot[memory]`，并提供运行时检测。
- `sentence-transformers` 往往会带来较重的依赖（如 torch）；若“超轻量”是硬约束，可增加轻依赖备选（ONNX/fastembed）或 remote embedding + 本地缓存。
- 降级链路建议：`Hybrid(BM25+Vector)` → `FTS-only` → `Markdown-only`（确保至少能工作）。

### 11.6 性能与并发（async 不阻塞）

- 索引与 embedding 建议放到后台线程/队列（`asyncio.to_thread` 或单独 worker），避免阻塞 AgentLoop。
- embedding/入库建议 batch 化（`embed_batch`），并对 observation 做截断与去重（减少向量写入量）。
- SQLite 建议开启 WAL，并将写入串行化（避免多协程并发写导致锁争用）。

### 11.7 安全与隐私（必须项）

- observation/summarizer 在写入/索引前做 redaction（API key、token、cookie、邮箱、手机号等），并提供 allowlist/denylist 配置。
- 对 `exec` 工具的观察默认更保守（只记录命令与少量输出片段），避免把敏感环境变量与完整输出进入长期记忆。

### 11.8 验收与基准（可复现）

- 规定基线（v1.0 或 “仅 Markdown”）与样本集（例如 30 条典型问题）。
- 统计口径：prompt tokens / completion tokens / 总 tokens、首字延迟、检索耗时、索引耗时。
- 给出通过阈值：例如“平均 prompt tokens 下降 ≥ 50%，同时相关召回命中率不下降”。

---

*Plan Status: Final v2.0*
*Ready for Implementation*
