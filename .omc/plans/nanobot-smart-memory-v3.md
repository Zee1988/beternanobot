# Nanobot 智能记忆系统 v3.0

> 状态: **Draft Plan**
> 创建时间: 2026-02-11
> 基于: v2.0 架构审查反馈
> 技术栈: 纯 Python (fastembed + SQLite FTS5)

---

## 一、v2.0 问题总结与 v3.0 解决方案

### 根本矛盾：依赖体积

| 问题 | v2.0 | v3.0 解决方案 |
|------|------|--------------|
| sentence-transformers 拉入 PyTorch ~2GB | 与"超轻量"定位矛盾 | 替换为 `fastembed`（ONNX Runtime，~150MB） |
| ChromaDB 依赖链过重（onnxruntime, fastapi, posthog） | 额外 ~200MB | 去掉 ChromaDB，用 SQLite 存向量（`sqlite-vec` 或纯 numpy） |
| 总安装体积 ~2-3GB | 膨胀 10-15 倍 | 控制在 ~200MB 以内 |

### P0 级缺陷修复

| # | v2.0 缺陷 | v3.0 修复 |
|---|----------|----------|
| 1 | 同步 embedding 阻塞 async event loop | 所有 embedding 调用包裹 `asyncio.to_thread()` |
| 2 | ChromaDB `add` 重复 ID 崩溃 | 统一使用 SQLite `INSERT OR REPLACE` |
| 3 | FTS5 `id` 列被分词索引 | 使用 content table 模式，id 存普通表 |
| 4 | FTS `INSERT` 无幂等保护 | 通过普通表 UNIQUE 约束 + trigger 同步 |
| 5 | Hook 失败中断主流程 | try/except 隔离，失败只记日志 |
| 6 | `HookManager` 缺 `emit_session_end` | 补齐所有 emit 方法 |

### P1 级问题修复

| # | v2.0 遗漏 | v3.0 修复 |
|---|----------|----------|
| 7 | `ContextBuilder` async 传播未考虑 | `get_memory_context` 保持同步，embedding 检索在 AgentLoop 层异步预取 |
| 8 | 分块策略缺失 | 按段落分块，max 256 tokens |
| 9 | doc_id 碰撞风险 | `{source_type}:{sha256[:16]}` 格式 |
| 10 | session end 触发时机未定义 | SessionManager 增加 idle timeout |
| 11 | `[-1000:]` 字符级截断 | 按段落边界截断 |
| 12 | 无向后兼容 | 运行时检测依赖，降级到原始 MemoryStore |
| 13 | SQLite 未开 WAL | 初始化时 `PRAGMA journal_mode=WAL` |
| 14 | FTS query 未 sanitize | 转义 FTS5 特殊字符 |

---

## 二、架构设计

### 2.1 核心决策：去掉 ChromaDB，统一用 SQLite

v3.0 最大的架构变化：**一个 SQLite 数据库同时承载向量索引和全文索引**。

理由：
- ChromaDB 内部也是 SQLite + onnxruntime，我们直接用 SQLite 省去中间层
- `sqlite-vec` 扩展提供原生向量搜索（余弦/L2），零额外依赖
- 一个 DB 文件 = 原子事务，解决 v2.0 的"三写非原子性"问题
- 备份/迁移只需复制一个文件

### 2.2 依赖选择

```
# 必选（~200MB 总计）
fastembed>=0.3.0          # ONNX embedding，无 PyTorch（~150MB）

# 可选（运行时检测）
sqlite-vec>=0.1.0         # SQLite 向量扩展（~5MB，纯 C）
```

降级链路：
```
Level 0: fastembed + sqlite-vec + FTS5  → 混合检索（最佳）
Level 1: fastembed + FTS5               → 向量用 numpy 暴力搜索（无 sqlite-vec 时）
Level 2: FTS5 only                      → 纯关键词检索（无 fastembed 时）
Level 3: Markdown only                  → 原始 MemoryStore（无任何额外依赖）
```

### 2.3 架构图

```
┌──────────────────────────────────────────────────────────┐
│              Nanobot Smart Memory v3.0                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  AgentLoop                                               │
│  ├─ _process_message()                                   │
│  │   ├─ memory_task = asyncio.to_thread(recall, query)   │
│  │   ├─ messages = context.build_messages(...)            │
│  │   ├─ memory_ctx = await memory_task  ← 并行预取       │
│  │   └─ inject memory_ctx into messages                  │
│  │                                                       │
│  └─ HookManager (fire-and-forget, isolated)              │
│      ├─ ObservationHook → index_queue                    │
│      └─ SummaryHook → index_queue                        │
│                                                          │
│  IndexWorker (后台线程)                                   │
│  └─ 消费 index_queue → chunk → embed → write SQLite      │
│                                                          │
│  MemoryStore (重构)                                       │
│  ├─ get_memory_context() → 同步，保持 ContextBuilder 兼容 │
│  ├─ recall(query) → 同步，在 to_thread 中调用             │
│  └─ Markdown 读写（不变）                                 │
│                                                          │
│  Storage Layer (单一 SQLite DB)                           │
│  ├─ docs 表 (id, source_type, text, embedding, metadata) │
│  ├─ docs_fts 虚拟表 (content table 模式)                  │
│  └─ manifest 表 (source_path, mtime, content_hash)       │
│                                                          │
│  EmbeddingService                                        │
│  ├─ fastembed (默认)                                     │
│  ├─ remote API (可选)                                    │
│  └─ None (降级)                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 三、文件结构

### 新增文件

```
nanobot/
├── agent/
│   ├── hooks.py           # HookManager + AgentHook ABC + ToolExecution
│   ├── observation.py     # ObservationHook（写入 index_queue）
│   ├── summarizer.py      # SummaryHook（会话摘要 → index_queue）
│   ├── memory.py          # MemoryStore 重构（保持同步接口 + recall 方法）
│   └── indexer.py         # IndexWorker 后台线程（消费 queue → chunk → embed → SQLite）
├── storage/
│   ├── embedding.py       # EmbeddingService（fastembed / remote / None）
│   ├── database.py        # MemoryDB（单一 SQLite：docs + FTS5 + manifest）
│   └── chunker.py         # 文本分块（按段落，max 256 tokens）
```

### 数据存储

```
workspace/
├── memory/                 # Markdown（不变，保持人类可读）
│   ├── MEMORY.md
│   └── 2026-02-11.md
├── .storage/
│   └── memory.db           # 单一 SQLite（向量 + FTS + manifest）
```

---

## 四、核心模块实现

### 4.1 EmbeddingService（运行时降级）

```python
# nanobot/storage/embedding.py

import asyncio
from typing import Protocol

class EmbeddingProvider(Protocol):
    """Embedding 提供者协议"""
    dimension: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...

class FastEmbedProvider:
    """基于 fastembed 的本地 ONNX embedding"""
    MODELS = {
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "chinese": "BAAI/bge-small-zh-v1.5",
        "english": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(self, model_name: str = "multilingual"):
        from fastembed import TextEmbedding
        actual = self.MODELS.get(model_name, model_name)
        self._model = TextEmbedding(model_name=actual)
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        sample = list(self._model.embed(["test"]))[0]
        return len(sample)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]


class EmbeddingService:
    """统一入口，运行时检测可用后端"""

    def __init__(self, model_name: str = "multilingual"):
        self._provider: EmbeddingProvider | None = None
        self._model_name = model_name
        self._init_provider()

    def _init_provider(self):
        # Level 0/1: 尝试 fastembed
        try:
            self._provider = FastEmbedProvider(self._model_name)
            return
        except ImportError:
            pass
        # Level 2/3: 无 embedding
        self._provider = None

    @property
    def available(self) -> bool:
        return self._provider is not None

    @property
    def dimension(self) -> int:
        return self._provider.dimension if self._provider else 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not self._provider:
            return []
        return self._provider.encode(texts)

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        """非阻塞 embedding — 在线程池中执行"""
        if not self._provider:
            return []
        return await asyncio.to_thread(self._provider.encode, texts)
```

### 4.2 统一 SQLite 存储（解决三写非原子性）

```python
# nanobot/storage/database.py

import sqlite3
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class DocRecord:
    id: str
    source_type: str          # daily_note | long_term | observation | summary
    text: str
    title: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    chunk_index: int = 0


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    source_type: str
    title: str = ""
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryDB:
    """单一 SQLite 数据库：docs + FTS5 + manifest + 可选向量"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init()

    def _init(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")

        # 主表：存储文档、embedding、metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                text TEXT NOT NULL,
                title TEXT DEFAULT '',
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                content_hash TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # FTS5 content table 模式：id 不被分词
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
            USING fts5(text, title, content=docs, content_rowid=rowid)
        """)

        # 触发器：docs 写入时自动同步 FTS
        for op, prefix in [("INSERT", "new"), ("DELETE", "old"),
                           ("UPDATE", "old")]:
            try:
                self.conn.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS docs_fts_{op.lower()}
                    AFTER {op} ON docs BEGIN
                        INSERT INTO docs_fts(docs_fts, rowid, text, title)
                        VALUES('delete', {prefix}.rowid, {prefix}.text,
                               {prefix}.title);
                    END
                """)
            except sqlite3.OperationalError:
                pass

        # UPDATE 还需要插入新值
        try:
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_fts_update_insert
                AFTER UPDATE ON docs BEGIN
                    INSERT INTO docs_fts(rowid, text, title)
                    VALUES(new.rowid, new.text, new.title);
                END
            """)
        except sqlite3.OperationalError:
            pass


        # Manifest 表：跟踪文件索引状态
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                source_path TEXT PRIMARY KEY,
                mtime REAL,
                content_hash TEXT,
                doc_ids TEXT DEFAULT '[]',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def upsert_doc(self, doc: DocRecord) -> None:
        """原子写入：docs + FTS 通过触发器自动同步"""
        embedding_blob = self._encode_vec(doc.embedding) if doc.embedding else None
        self.conn.execute("""
            INSERT OR REPLACE INTO docs
                (id, source_type, text, title, embedding, metadata,
                 content_hash, chunk_index, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            doc.id, doc.source_type, doc.text, doc.title,
            embedding_blob, json.dumps(doc.metadata),
            doc.content_hash, doc.chunk_index,
        ))
        self.conn.commit()

    def delete_by_source(self, source_path: str) -> None:
        """删除某文件的所有 chunks"""
        row = self.conn.execute(
            "SELECT doc_ids FROM manifest WHERE source_path = ?",
            (source_path,)
        ).fetchone()
        if row:
            doc_ids = json.loads(row[0])
            for did in doc_ids:
                self.conn.execute("DELETE FROM docs WHERE id = ?", (did,))
            self.conn.execute(
                "DELETE FROM manifest WHERE source_path = ?",
                (source_path,)
            )
            self.conn.commit()

    def search_fts(self, query: str, limit: int = 10) -> list[SearchResult]:
        """全文搜索（BM25）"""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        cursor = self.conn.execute("""
            SELECT d.id, d.text, bm25(docs_fts) as score,
                   d.source_type, d.title,
                   snippet(docs_fts, 0, '[', ']', '...', 12) as snip
            FROM docs_fts f
            JOIN docs d ON d.rowid = f.rowid
            WHERE docs_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (safe_query, limit))
        return [
            SearchResult(id=r[0], text=r[1], score=r[2],
                         source_type=r[3], title=r[4], snippet=r[5])
            for r in cursor.fetchall()
        ]

    def search_vector(self, query_vec: list[float], limit: int = 10
                      ) -> list[SearchResult]:
        """向量搜索（余弦相似度，暴力扫描 — 文档量 <10K 时足够快）"""
        import struct
        rows = self.conn.execute(
            "SELECT id, text, embedding, source_type, title FROM docs "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []

        scored = []
        for row_id, text, emb_blob, src_type, title in rows:
            if not emb_blob:
                continue
            vec = self._decode_vec(emb_blob)
            sim = self._cosine_sim(query_vec, vec)
            scored.append(SearchResult(
                id=row_id, text=text, score=sim,
                source_type=src_type, title=title,
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]


    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """转义 FTS5 特殊字符，防止语法错误"""
        import re
        # 移除 FTS5 操作符
        cleaned = re.sub(r'\b(AND|OR|NOT|NEAR)\b', '', query, flags=re.IGNORECASE)
        # 转义特殊字符
        cleaned = cleaned.replace('"', '').replace('*', '').replace('(', '').replace(')', '')
        cleaned = cleaned.strip()
        if not cleaned:
            return ""
        # 用双引号包裹每个词，确保安全
        words = cleaned.split()
        return " ".join(f'"{w}"' for w in words if w)

    @staticmethod
    def _encode_vec(vec: list[float]) -> bytes:
        import struct
        return struct.pack(f'{len(vec)}f', *vec)

    @staticmethod
    def _decode_vec(blob: bytes) -> list[float]:
        import struct
        n = len(blob) // 4
        return list(struct.unpack(f'{n}f', blob))

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self):
        self.conn.close()
```

### 4.3 文本分块

```python
# nanobot/storage/chunker.py

import hashlib
import re

def chunk_text(
    text: str,
    source_type: str,
    source_id: str = "",
    max_chars: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """
    按段落分块，保证每块 <= max_chars。
    返回 [{id, text, chunk_index, content_hash}, ...]
    """
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current)
            # 保留 overlap
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = current + "\n\n" + para if current else para

    if current:
        chunks.append(current)

    results = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
        doc_id = f"{source_type}:{source_id}:{i}" if source_id else f"{source_type}:{content_hash}:{i}"
        results.append({
            "id": doc_id,
            "text": chunk,
            "chunk_index": i,
            "content_hash": content_hash,
        })
    return results
```


### 4.4 混合检索（RRF 融合）

```python
# nanobot/agent/retrieval.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievedChunk:
    id: str
    text: str
    fused_score: float = 0.0
    source_type: str = ""
    title: str = ""
    snippet: str | None = None
    is_full: bool = False


class HybridRetrieval:
    """混合检索：RRF 融合向量 + BM25 结果"""

    def __init__(self, db: "MemoryDB", embedding: "EmbeddingService"):
        self.db = db
        self.embedding = embedding

    def search(self, query: str, limit: int = 8, rrf_k: int = 60
               ) -> list[RetrievedChunk]:
        # 1) FTS5 搜索（始终可用）
        fts_results = self.db.search_fts(query, limit * 2)

        # 2) 向量搜索（仅当 embedding 可用时）
        vec_results = []
        if self.embedding.available:
            vecs = self.embedding.encode([query])
            if vecs:
                vec_results = self.db.search_vector(vecs[0], limit * 2)

        # 3) RRF 融合
        scores: dict[str, float] = {}
        text_map: dict[str, "SearchResult"] = {}

        for rank, r in enumerate(vec_results, 1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank)
            text_map[r.id] = r

        for rank, r in enumerate(fts_results, 1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank)
            if r.id not in text_map:
                text_map[r.id] = r

        # 4) 排序并构建结果
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ordered[:limit]:
            r = text_map[doc_id]
            results.append(RetrievedChunk(
                id=doc_id, text=r.text, fused_score=score,
                source_type=r.source_type, title=r.title,
                snippet=getattr(r, 'snippet', None),
            ))
        return results

    def progressive_retrieve(
        self, query: str,
        max_full: int = 3, max_snippets: int = 5,
    ) -> list[RetrievedChunk]:
        """渐进检索：top-k 全文 + 后续 snippet"""
        results = self.search(query, limit=max_full + max_snippets)
        for r in results[:max_full]:
            r.is_full = True
        return results

    def format_context(self, results: list[RetrievedChunk],
                       max_chars: int = 3000) -> str:
        """格式化检索结果为 prompt 上下文"""
        parts = []
        total = 0
        for r in results:
            if r.is_full:
                text = r.text
            else:
                text = r.snippet or r.text[:200] + "..."
            entry = f"### {r.title or r.source_type}\n{text}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)
```


### 4.5 Hooks 系统（失败隔离）

```python
# nanobot/agent/hooks.py

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class ToolExecution:
    tool_name: str
    arguments: dict[str, Any]
    result: str
    error: Exception | None
    timestamp: datetime
    duration_ms: float
    session_key: str


class AgentHook(ABC):
    async def on_message_received(self, session_key: str, content: str) -> None:
        pass

    async def on_tool_executed(self, execution: ToolExecution) -> None:
        pass

    async def on_session_idle(self, session_key: str,
                              history: list[dict], turn_count: int) -> None:
        """会话空闲时触发（替代模糊的 session_end）"""
        pass


class HookManager:
    def __init__(self):
        self._hooks: list[AgentHook] = []

    def register(self, hook: AgentHook) -> None:
        self._hooks.append(hook)

    async def emit_tool_executed(self, execution: ToolExecution) -> None:
        for hook in self._hooks:
            try:
                await hook.on_tool_executed(execution)
            except Exception as e:
                logger.warning(f"Hook {hook.__class__.__name__} failed: {e}")

    async def emit_session_idle(self, session_key: str,
                                history: list[dict],
                                turn_count: int) -> None:
        for hook in self._hooks:
            try:
                await hook.on_session_idle(session_key, history, turn_count)
            except Exception as e:
                logger.warning(f"Hook {hook.__class__.__name__} failed: {e}")

    async def emit_message_received(self, session_key: str,
                                    content: str) -> None:
        for hook in self._hooks:
            try:
                await hook.on_message_received(session_key, content)
            except Exception as e:
                logger.warning(f"Hook {hook.__class__.__name__} failed: {e}")
```

### 4.6 IndexWorker（后台线程，不阻塞 event loop）

```python
# nanobot/agent/indexer.py

import asyncio
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

@dataclass
class IndexTask:
    text: str
    source_type: str
    title: str = ""
    source_id: str = ""
    metadata: dict[str, Any] | None = None


class IndexWorker:
    """后台线程消费索引任务，避免阻塞 async event loop"""

    def __init__(self, db: "MemoryDB", embedding: "EmbeddingService"):
        self.db = db
        self.embedding = embedding
        self._queue: queue.Queue[IndexTask | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, task: IndexTask):
        self._queue.put(task)

    def stop(self):
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        from nanobot.storage.chunker import chunk_text
        from nanobot.storage.database import DocRecord

        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                chunks = chunk_text(
                    task.text, task.source_type, task.source_id
                )
                # Batch embed
                texts = [c["text"] for c in chunks]
                embeddings = self.embedding.encode(texts) if self.embedding.available else [[] for _ in texts]

                for chunk, emb in zip(chunks, embeddings):
                    self.db.upsert_doc(DocRecord(
                        id=chunk["id"],
                        source_type=task.source_type,
                        text=chunk["text"],
                        title=task.title,
                        embedding=emb,
                        metadata=task.metadata or {},
                        content_hash=chunk["content_hash"],
                        chunk_index=chunk["chunk_index"],
                    ))
            except Exception as e:
                logger.warning(f"IndexWorker failed for {task.source_type}: {e}")
```


### 4.7 ObservationHook（脱敏 + 提交到 IndexWorker）

```python
# nanobot/agent/observation.py

import re
from nanobot.agent.hooks import AgentHook, ToolExecution

SIGNIFICANT_TOOLS = {"write_file", "edit_file", "exec"}

# 脱敏正则
REDACT_PATTERNS = [
    (re.compile(r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S+', re.I), '[REDACTED]'),
    (re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), '[EMAIL]'),
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), '[API_KEY]'),
]


def redact(text: str) -> str:
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class ObservationHook(AgentHook):
    def __init__(self, indexer: "IndexWorker", memory_dir: "Path"):
        self.indexer = indexer
        self.memory_dir = memory_dir

    async def on_tool_executed(self, execution: ToolExecution) -> None:
        if execution.tool_name not in SIGNIFICANT_TOOLS:
            return
        if execution.error:
            return

        # 格式化 + 脱敏
        result_preview = redact(execution.result[:500])
        args_preview = redact(str(execution.arguments)[:300])

        content = (
            f"## {execution.tool_name} @ {execution.timestamp:%H:%M}\n\n"
            f"**Args:** {args_preview}\n"
            f"**Result:** {result_preview}\n"
        )

        # 1. 写入 Markdown（人类可读）
        today_file = self.memory_dir / f"{execution.timestamp:%Y-%m-%d}.md"
        with open(today_file, "a", encoding="utf-8") as f:
            f.write(content + "\n---\n\n")

        # 2. 提交到 IndexWorker（非阻塞）
        from nanobot.agent.indexer import IndexTask
        self.indexer.submit(IndexTask(
            text=content,
            source_type="observation",
            title=execution.tool_name,
            source_id=f"{execution.session_key}:{execution.timestamp:%H%M%S}",
        ))
```

### 4.8 SummaryHook（idle 触发，非每条消息）

```python
# nanobot/agent/summarizer.py

from nanobot.agent.hooks import AgentHook

SUMMARY_PROMPT = """请用中文简要总结这段对话（200字以内）：
1. 用户想做什么
2. 采取了哪些关键操作
3. 最终结果

{conversation}
"""


class SummaryHook(AgentHook):
    def __init__(self, indexer: "IndexWorker", memory_dir: "Path",
                 provider: "LLMProvider", min_turns: int = 3):
        self.indexer = indexer
        self.memory_dir = memory_dir
        self.provider = provider
        self.min_turns = min_turns

    async def on_session_idle(self, session_key: str,
                              history: list[dict],
                              turn_count: int) -> None:
        """会话空闲时生成摘要（而非每条消息后）"""
        user_msgs = [m for m in history if m.get("role") == "user"]
        if len(user_msgs) < self.min_turns:
            return

        conv = "\n".join(
            f"{m['role']}: {m.get('content', '')[:500]}"
            for m in history[-20:]  # 最近 20 条
        )
        resp = await self.provider.chat([{
            "role": "user",
            "content": SUMMARY_PROMPT.format(conversation=conv),
        }], max_tokens=300)

        summary = resp.content or ""
        if not summary:
            return

        # 写入 MEMORY.md
        memory_file = self.memory_dir / "MEMORY.md"
        from datetime import datetime
        entry = f"\n\n## {datetime.now():%Y-%m-%d %H:%M} - {session_key}\n{summary}\n"
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(entry)

        # 提交索引
        from nanobot.agent.indexer import IndexTask
        self.indexer.submit(IndexTask(
            text=summary,
            source_type="summary",
            title=f"Session: {session_key}",
            source_id=session_key,
        ))
```


### 4.9 MemoryStore 重构（保持同步接口 + 新增 recall）

```python
# nanobot/agent/memory.py（重构）

from pathlib import Path
from datetime import datetime

from nanobot.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """
    记忆系统 v3：保持原有同步接口不变，新增 recall() 用于语义检索。
    ContextBuilder 继续调用 get_memory_context()（同步）。
    AgentLoop 在 asyncio.to_thread 中调用 recall()（不阻塞）。
    """

    def __init__(self, workspace: Path, embedding_model: str = "multilingual"):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self._embedding_model = embedding_model
        self._db = None
        self._embedding = None
        self._retrieval = None
        self._indexer = None

    # --- 原有接口（完全不变）---

    def get_today_file(self) -> Path:
        return self.memory_dir / f"{today_date()}.md"

    def read_today(self) -> str:
        today_file = self.get_today_file()
        return today_file.read_text(encoding="utf-8") if today_file.exists() else ""

    def append_today(self, content: str) -> None:
        today_file = self.get_today_file()
        if today_file.exists():
            content = today_file.read_text(encoding="utf-8") + "\n" + content
        else:
            content = f"# {today_date()}\n\n" + content
        today_file.write_text(content, encoding="utf-8")

    def read_long_term(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        """同步接口，ContextBuilder 直接调用（不变）"""
        parts = []
        long_term = self.read_long_term()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        today = self.read_today()
        if today:
            parts.append("## Today's Notes\n" + today)
        return "\n\n".join(parts) if parts else ""

    # --- 新增：语义检索（同步方法，在 to_thread 中调用）---

    def recall(self, query: str, max_full: int = 3,
               max_snippets: int = 5) -> str:
        """语义检索记忆，返回格式化的上下文字符串。同步方法。"""
        retrieval = self._get_retrieval()
        if not retrieval:
            return ""
        results = retrieval.progressive_retrieve(
            query, max_full=max_full, max_snippets=max_snippets
        )
        if not results:
            return ""
        return retrieval.format_context(results)

    # --- 懒初始化 ---

    def _get_retrieval(self):
        if self._retrieval is not None:
            return self._retrieval
        try:
            from nanobot.storage.embedding import EmbeddingService
            from nanobot.storage.database import MemoryDB
            from nanobot.agent.retrieval import HybridRetrieval

            db_path = self.workspace / ".storage" / "memory.db"
            self._db = MemoryDB(db_path)
            self._embedding = EmbeddingService(self._embedding_model)
            self._retrieval = HybridRetrieval(self._db, self._embedding)
            self._index_existing_files()
            return self._retrieval
        except ImportError:
            return None

    def _get_indexer(self):
        if self._indexer is not None:
            return self._indexer
        retrieval = self._get_retrieval()
        if not retrieval:
            return None
        try:
            from nanobot.agent.indexer import IndexWorker
            self._indexer = IndexWorker(self._db, self._embedding)
            self._indexer.start()
            return self._indexer
        except ImportError:
            return None

    def _index_existing_files(self):
        """增量索引：只索引 mtime 变化的文件"""
        if not self._db:
            return
        import json
        from nanobot.storage.chunker import chunk_text
        from nanobot.storage.database import DocRecord

        for md_file in sorted(self.memory_dir.glob("*.md")):
            stat = md_file.stat()
            source_path = str(md_file)
            row = self._db.conn.execute(
                "SELECT mtime FROM manifest WHERE source_path = ?",
                (source_path,)
            ).fetchone()
            if row and abs(row[0] - stat.st_mtime) < 0.01:
                continue  # 未变化，跳过

            # 删除旧 chunks
            self._db.delete_by_source(source_path)

            # 重新分块 + 索引
            text = md_file.read_text(encoding="utf-8")
            source_type = "long_term" if md_file.name == "MEMORY.md" else "daily_note"
            chunks = chunk_text(text, source_type, md_file.stem)

            texts = [c["text"] for c in chunks]
            embeddings = (self._embedding.encode(texts)
                          if self._embedding and self._embedding.available
                          else [[] for _ in texts])

            doc_ids = []
            for chunk, emb in zip(chunks, embeddings):
                self._db.upsert_doc(DocRecord(
                    id=chunk["id"],
                    source_type=source_type,
                    text=chunk["text"],
                    title=md_file.name,
                    embedding=emb,
                    content_hash=chunk["content_hash"],
                    chunk_index=chunk["chunk_index"],
                ))
                doc_ids.append(chunk["id"])

            # 更新 manifest
            self._db.conn.execute("""
                INSERT OR REPLACE INTO manifest (source_path, mtime, content_hash, doc_ids)
                VALUES (?, ?, ?, ?)
            """, (source_path, stat.st_mtime, "", json.dumps(doc_ids)))
            self._db.conn.commit()
```


---

## 五、AgentLoop 集成（最小侵入）

关键设计：`ContextBuilder.build_system_prompt()` 保持同步不变。语义检索在 `AgentLoop._process_message()` 中通过 `asyncio.to_thread` 并行预取，然后注入到 messages 中。

```python
# nanobot/agent/loop.py（修改部分）

import time
from datetime import datetime
from nanobot.agent.hooks import HookManager, ToolExecution

class AgentLoop:
    def __init__(self, ..., memory_config: "MemoryConfig | None" = None):
        # 现有代码不变...

        # 新增：Hooks
        self.hooks = HookManager()

        # 新增：配置记忆增强
        if memory_config and memory_config.enabled:
            indexer = self.context.memory._get_indexer()
            if indexer:
                if memory_config.enable_observations:
                    from nanobot.agent.observation import ObservationHook
                    self.hooks.register(ObservationHook(indexer, self.workspace / "memory"))
                if memory_config.enable_summaries:
                    from nanobot.agent.summarizer import SummaryHook
                    self.hooks.register(SummaryHook(
                        indexer, self.workspace / "memory",
                        self.provider, memory_config.summary_min_turns,
                    ))

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        # ... 现有代码到 build_messages 之前 ...

        # 新增：并行预取语义记忆（不阻塞 context 构建）
        recall_task = asyncio.create_task(
            asyncio.to_thread(self.context.memory.recall, msg.content)
        )

        # 构建 messages（同步，不受影响）
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        # 等待语义记忆结果，注入到 system prompt 后
        recall_ctx = await recall_task
        if recall_ctx:
            # 在 system message 后插入记忆上下文
            memory_msg = {
                "role": "system",
                "content": f"# Relevant Memories\n\n{recall_ctx}",
            }
            messages.insert(1, memory_msg)

        # Agent loop（现有逻辑）
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )

            if response.has_tool_calls:
                # ... 现有 tool call 处理 ...

                for tool_call in response.tool_calls:
                    start = time.time()
                    result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    duration = (time.time() - start) * 1000

                    # 新增：触发 Hook（fire-and-forget）
                    await self.hooks.emit_tool_executed(ToolExecution(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        result=result,
                        error=None,
                        timestamp=datetime.now(),
                        duration_ms=duration,
                        session_key=msg.session_key,
                    ))

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        # ... 现有保存逻辑 ...

        # 新增：通知 session idle（用于摘要生成）
        user_turns = len([m for m in session.messages if m.get("role") == "user"])
        await self.hooks.emit_session_idle(
            msg.session_key, session.get_history(), user_turns
        )

        return OutboundMessage(...)
```


---

## 六、配置（纳入现有 config.json）

### 6.1 Schema 新增

```python
# nanobot/config/schema.py 新增

class MemoryRetrievalConfig(BaseModel):
    limit: int = 8
    max_full_docs: int = 3
    max_snippets: int = 5
    rrf_k: int = 60
    max_context_chars: int = 3000

class MemoryConfig(BaseModel):
    enabled: bool = False           # 默认关闭，需显式开启
    embedding_model: str = "multilingual"
    enable_observations: bool = True
    enable_summaries: bool = True
    summary_min_turns: int = 3
    retrieval: MemoryRetrievalConfig = Field(default_factory=MemoryRetrievalConfig)

class Config(BaseSettings):
    # ... 现有字段 ...
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
```

### 6.2 config.json 示例

```json
{
  "memory": {
    "enabled": true,
    "embeddingModel": "multilingual",
    "enableObservations": true,
    "enableSummaries": true,
    "summaryMinTurns": 3,
    "retrieval": {
      "limit": 8,
      "maxFullDocs": 3,
      "maxSnippets": 5,
      "rrfK": 60,
      "maxContextChars": 3000
    }
  }
}
```

通过现有 `loader.py` 的 `convert_keys` (camelCase → snake_case) 自动转换，无需额外迁移逻辑。

---

## 七、pyproject.toml 变更

```toml
[project.optional-dependencies]
memory = [
    "fastembed>=0.3.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "fastembed>=0.3.0",
]
```

安装方式：
```bash
pip install nanobot-ai[memory]    # 带记忆增强
pip install nanobot-ai            # 不带（原始 Markdown 记忆）
```

---

## 八、v2.0 vs v3.0 对比

| 维度 | v2.0 | v3.0 |
|------|------|------|
| Embedding | sentence-transformers (~2GB) | fastembed (~150MB) |
| 向量存储 | ChromaDB (额外 ~200MB) | SQLite + struct.pack (0 额外依赖) |
| 全文搜索 | 独立 SQLite FTS5 | 同一 SQLite，content table 模式 |
| 原子性 | 三写非原子（Chroma + FTS + Markdown） | 单 SQLite 事务 + Markdown |
| async 安全 | 同步 embedding 阻塞 event loop | asyncio.to_thread + IndexWorker 后台线程 |
| Hook 隔离 | 无（失败中断主流程） | try/except 隔离 |
| 分块 | 无 | 按段落，max 800 chars |
| 幂等 | add 重复崩溃 | INSERT OR REPLACE |
| 降级 | 无 | 4 级降级链路 |
| session end | 未定义 | idle 触发（每次消息处理后） |
| 脱敏 | 无 | 正则 redaction |
| 安装体积 | ~2-3GB | ~200MB |
| 新增代码量 | ~8 文件 | ~8 文件（更精简） |


---

## 九、实施计划

| Phase | 内容 | 文件 | 时间 |
|-------|------|------|------|
| 1 | 存储层 | `storage/embedding.py`, `storage/database.py`, `storage/chunker.py` | 1 天 |
| 2 | 检索层 | `agent/retrieval.py`, `agent/memory.py` 重构 | 1 天 |
| 3 | Hooks + Indexer | `agent/hooks.py`, `agent/observation.py`, `agent/summarizer.py`, `agent/indexer.py` | 1 天 |
| 4 | 集成 + 配置 | `agent/loop.py` 修改, `config/schema.py` 新增, `pyproject.toml` | 0.5 天 |
| 5 | 测试 + 验收 | 单元测试, 集成测试, 基准对比 | 1.5 天 |

**总计：5 天**

### Phase 详细

**Phase 1: 存储层**
- 实现 `EmbeddingService`，验证 fastembed 加载和 encode
- 实现 `MemoryDB`，验证 docs + FTS5 content table + manifest + 触发器
- 实现 `chunk_text`，验证分块逻辑
- 测试：embedding encode/decode、SQLite upsert/search/delete、FTS sanitize

**Phase 2: 检索层**
- 实现 `HybridRetrieval`，验证 RRF 融合
- 重构 `MemoryStore`，保持原有接口 + 新增 `recall()`
- 验证降级链路（有/无 fastembed 两种场景）
- 测试：recall 返回相关结果、降级到纯 Markdown

**Phase 3: Hooks + Indexer**
- 实现 `HookManager`（失败隔离）
- 实现 `IndexWorker`（后台线程）
- 实现 `ObservationHook`（脱敏 + 提交）
- 实现 `SummaryHook`（idle 触发）
- 测试：Hook 失败不影响主流程、IndexWorker 正确消费队列

**Phase 4: 集成 + 配置**
- `config/schema.py` 新增 `MemoryConfig`
- `agent/loop.py` 集成：并行预取 + Hook 触发
- `pyproject.toml` 新增 `[memory]` optional dependency
- 端到端测试：发消息 → 检索 → 观察记录 → 摘要生成

**Phase 5: 测试 + 验收**
- 基准测试：对比 v1（纯 Markdown）的 prompt token 消耗
- 检索质量：30 条典型问题的命中率
- 性能：embedding 延迟、索引延迟、检索延迟
- 降级验证：卸载 fastembed 后功能正常

---

## 十、验收标准

- [ ] `pip install nanobot-ai[memory]` 成功安装 fastembed
- [ ] `pip install nanobot-ai`（无 memory extra）正常运行，降级到纯 Markdown
- [ ] fastembed 模型首次自动下载并加载（~150MB）
- [ ] 单一 SQLite 文件包含 docs + FTS + manifest
- [ ] `INSERT OR REPLACE` 幂等：重复索引不膨胀
- [ ] 文件删除/更新触发索引同步（通过 manifest mtime 检测）
- [ ] 混合检索 RRF 融合正确（向量 + BM25）
- [ ] FTS query 特殊字符不导致崩溃
- [ ] Hook 失败只记日志，不中断主流程
- [ ] embedding 调用不阻塞 async event loop（to_thread）
- [ ] IndexWorker 后台线程正确消费队列
- [ ] 观察记录自动脱敏（API key、email 等）
- [ ] 会话摘要在 idle 时生成（非每条消息）
- [ ] `ContextBuilder.build_system_prompt()` 保持同步（无 async 传播）
- [ ] 4 级降级链路全部可用
- [ ] Prompt token 消耗降低 ≥ 40%（基于 30 条样本）
- [ ] 检索命中率 ≥ 70%（基于 30 条典型问题）

---

*Plan Status: v3.0 Draft*
*基于 v2.0 架构审查的全面修订*
