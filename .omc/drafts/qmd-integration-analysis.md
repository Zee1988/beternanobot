# QMD 集成 Nanobot Memory 架构分析

> 生成时间: 2026-02-11
> 状态: Draft - 待评审

---

## 一、架构对比分析

### 1.1 Nanobot 当前 Memory 架构

```
nanobot/agent/memory.py - MemoryStore
├── 存储结构
│   ├── MEMORY.md          # 长期记忆（手动管理）
│   └── memory/YYYY-MM-DD.md  # 每日笔记（自动按日期）
├── 检索方式
│   ├── read_today()       # 读取今日笔记
│   ├── read_long_term()   # 读取长期记忆
│   └── get_recent_memories(days=7)  # 获取最近N天
└── 上下文构建
    └── get_memory_context()  # 简单拼接返回
```

**特点:**
- ✅ 简单可靠，无外部依赖
- ✅ 人类可读的 Markdown 文件
- ✅ 零配置开箱即用
- ❌ 无语义搜索能力
- ❌ 线性检索，无法处理大量历史
- ❌ Token 浪费（总是加载完整内容）

### 1.2 QMD 架构

```
QMD Hybrid Search Pipeline
├── 索引层
│   ├── SQLite FTS5 (BM25 全文检索)
│   ├── sqlite-vec (向量索引)
│   └── content_vectors (800 token 分块)
├── 检索层
│   ├── search     # BM25 关键词搜索
│   ├── vsearch    # 向量语义搜索
│   └── query      # 混合搜索 + 重排序
├── 模型层 (本地 GGUF)
│   ├── embeddinggemma-300M  # 向量嵌入
│   ├── qwen3-reranker-0.6b  # 重排序
│   └── qmd-query-expansion   # 查询扩展
└── 接口层
    ├── CLI 命令
    └── MCP Server
```

**特点:**
- ✅ 混合检索（BM25 + 向量 + 重排序）
- ✅ 本地运行，隐私安全
- ✅ MCP 协议支持
- ✅ 智能分块和上下文感知
- ❌ 需要 Bun 运行时
- ❌ 需要下载 ~2GB 模型
- ❌ 需要 GPU/Metal 加速

---

## 二、集成可行性评估

### 2.1 技术可行性: ✅ 高

| 维度 | 评估 | 说明 |
|------|------|------|
| 语言兼容 | ✅ | QMD 是 CLI/MCP，Python 可通过子进程或 MCP 调用 |
| 数据格式 | ✅ | 两者都基于 Markdown，完全兼容 |
| 架构耦合 | ✅ | QMD 独立运行，nanobot 仅需调用接口 |
| 渐进迁移 | ✅ | 可保留原有文件结构，仅添加索引层 |

### 2.2 集成方式选择

#### 方案 A: MCP 协议集成（推荐）

```python
# nanobot 作为 MCP Client 调用 qmd
class QMDMemory:
    async def search(self, query: str) -> list[Document]:
        # 通过 MCP 调用 qmd_search / qmd_deep_search
        pass
```

**优点:**
- 标准化协议，未来可扩展
- qmd 独立进程，故障隔离
- 支持 HTTP 模式，模型常驻内存

**缺点:**
- 需要额外的 MCP client 实现
- 进程间通信开销

#### 方案 B: CLI 子进程调用

```python
# 直接执行 qmd 命令
class QMDMemory:
    async def search(self, query: str) -> list[Document]:
        result = await asyncio.create_subprocess_exec(
            "qmd", "search", query, "--json",
            stdout=asyncio.subprocess.PIPE
        )
        return json.loads(await result.stdout.read())
```

**优点:**
- 实现简单，无需 MCP 客户端
- 直接复用 qmd 的所有功能

**缺点:**
- 每次调用需要启动进程
- 冷启动慢（模型加载）

#### 方案 C: 混合模式（推荐实施路径）

1. 先用 CLI 方式快速验证
2. 后续升级为 MCP HTTP 模式
3. qmd daemon 常驻，避免冷启动

---

## 三、优缺点分析

### 3.1 采用 QMD 的优势

| 优势 | 影响 | 量化收益 |
|------|------|----------|
| **语义检索** | 根据含义而非关键词查找记忆 | 召回率提升 3-5x |
| **Token 节省** | 只检索相关片段而非全部 | 减少 60-80% context 消耗 |
| **长期记忆** | 可索引数月/数年的历史 | 突破 7 天限制 |
| **多源整合** | 可索引笔记、文档、会议记录 | 统一知识检索 |
| **本地隐私** | 所有计算在本地完成 | 满足数据安全要求 |
| **MCP 生态** | 与其他 MCP 工具互操作 | 生态扩展性 |

### 3.2 采用 QMD 的劣势

| 劣势 | 风险 | 缓解措施 |
|------|------|----------|
| **依赖复杂度** | 需要 Bun + 本地模型 | 提供一键安装脚本 |
| **资源占用** | ~2GB 模型 + GPU/内存 | 支持降级到纯 BM25 模式 |
| **冷启动慢** | 首次加载模型需 10-30s | 使用 daemon 模式常驻 |
| **平台限制** | macOS 最佳支持 | 测试 Linux/Windows 兼容性 |
| **维护成本** | 依赖外部项目更新 | 锁定版本 + 容错降级 |

### 3.3 不采用 QMD 的替代方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **保持现状** | 零成本，稳定 | 无法扩展 |
| **自建向量检索** | 完全可控 | 开发成本高 |
| **Chroma/Qdrant** | 成熟的向量库 | 无 BM25 混合检索 |
| **Elasticsearch** | 功能强大 | 太重，部署复杂 |

**结论:** QMD 是目前最适合 nanobot 场景的选择（本地、Markdown、混合检索、MCP）。

---

## 四、实现难点及解决方案

### 4.1 难点一：依赖安装

**问题:** 需要安装 Bun、qmd、下载模型

**解决方案:**

```python
# nanobot/utils/qmd_setup.py

async def ensure_qmd_installed():
    """检查并安装 qmd 依赖"""
    # 1. 检查 bun
    if not shutil.which("bun"):
        logger.info("Installing Bun...")
        await run("curl -fsSL https://bun.sh/install | bash")

    # 2. 检查 qmd
    if not shutil.which("qmd"):
        logger.info("Installing QMD...")
        await run("bun install -g github:tobi/qmd")

    # 3. 模型会在首次使用时自动下载
    return True
```

### 4.2 难点二：数据迁移

**问题:** 现有 MEMORY.md 和 daily notes 需要被索引

**解决方案:**

```python
# nanobot/agent/memory.py

class HybridMemoryStore:
    """混合存储：保留原有文件结构 + 添加 qmd 索引"""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.legacy = MemoryStore(workspace)  # 保留旧实现
        self.qmd_enabled = self._check_qmd()

    async def initialize(self):
        """初始化 qmd 集合"""
        if not self.qmd_enabled:
            return

        memory_dir = self.workspace / "memory"

        # 创建 qmd 集合
        await self._run_qmd(
            "collection", "add", str(memory_dir),
            "--name", "memory",
            "--mask", "**/*.md"
        )

        # 添加上下文描述
        await self._run_qmd(
            "context", "add", "qmd://memory",
            "Agent memory and daily notes"
        )

        # 生成向量嵌入
        await self._run_qmd("embed")
```

### 4.3 难点三：实时同步

**问题:** 新写入的记忆需要被立即索引

**解决方案:**

```python
class HybridMemoryStore:
    async def append_today(self, content: str) -> None:
        """追加今日笔记并更新索引"""
        # 1. 写入文件（保留原逻辑）
        self.legacy.append_today(content)

        # 2. 增量更新索引
        if self.qmd_enabled:
            await self._run_qmd("update")
            # 可选：只对修改的文件重新嵌入
            today_file = self.legacy.get_today_file()
            await self._run_qmd("embed", str(today_file))
```

### 4.4 难点四：检索整合

**问题:** 需要将 qmd 结果转换为 nanobot context 格式

**解决方案:**

```python
class HybridMemoryStore:
    async def get_memory_context(self, query: str = None) -> str:
        """获取记忆上下文，支持语义检索"""

        if not self.qmd_enabled or not query:
            # 降级到原有逻辑
            return self.legacy.get_memory_context()

        # 使用 qmd 语义检索
        result = await self._run_qmd(
            "query", query,
            "-n", "5",
            "--min-score", "0.3",
            "--json"
        )

        docs = json.loads(result)

        parts = []
        for doc in docs:
            parts.append(f"## {doc['title']} (score: {doc['score']:.0%})\n{doc['snippet']}")

        return "# Relevant Memories\n\n" + "\n\n---\n\n".join(parts)
```

### 4.5 难点五：降级策略

**问题:** qmd 不可用时（未安装、故障）需要平滑降级

**解决方案:**

```python
class HybridMemoryStore:
    def _check_qmd(self) -> bool:
        """检查 qmd 是否可用"""
        try:
            result = subprocess.run(
                ["qmd", "status"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    async def search(self, query: str) -> list[MemoryChunk]:
        """搜索记忆，自动降级"""
        if self.qmd_enabled:
            try:
                return await self._qmd_search(query)
            except Exception as e:
                logger.warning(f"QMD search failed, falling back: {e}")

        # 降级：简单关键词匹配
        return self._simple_search(query)

    def _simple_search(self, query: str) -> list[MemoryChunk]:
        """简单的文本搜索降级实现"""
        keywords = query.lower().split()
        results = []

        for file in self.legacy.list_memory_files():
            content = file.read_text()
            score = sum(1 for k in keywords if k in content.lower())
            if score > 0:
                results.append(MemoryChunk(
                    path=str(file),
                    content=content[:500],
                    score=score / len(keywords)
                ))

        return sorted(results, key=lambda x: x.score, reverse=True)[:5]
```

### 4.6 难点六：性能优化

**问题:** qmd daemon 管理和冷启动

**解决方案:**

```python
# nanobot/__main__.py

async def start_qmd_daemon():
    """启动 qmd MCP daemon"""
    await asyncio.create_subprocess_exec(
        "qmd", "mcp", "--http", "--daemon",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    logger.info("QMD daemon started on port 8181")

async def stop_qmd_daemon():
    """停止 qmd daemon"""
    await asyncio.create_subprocess_exec("qmd", "mcp", "stop")
```

---

## 五、实施路线图

### Phase 1: 基础集成（1-2 天）

- [ ] 创建 `HybridMemoryStore` 类
- [ ] 实现 qmd CLI 调用封装
- [ ] 添加依赖检测和降级逻辑
- [ ] 编写安装检查脚本

### Phase 2: 索引管理（1 天）

- [ ] 实现 memory 目录自动索引
- [ ] 实现增量更新机制
- [ ] 添加 context 描述配置

### Phase 3: 检索整合（1 天）

- [ ] 替换 `get_memory_context()` 实现
- [ ] 在 `ContextBuilder` 中使用语义检索
- [ ] 实现查询意图识别（何时用检索 vs 全量）

### Phase 4: 优化完善（1-2 天）

- [ ] 添加 daemon 模式支持
- [ ] 实现 MCP HTTP client
- [ ] 性能测试和调优
- [ ] 文档和配置说明

---

## 六、关键代码示例

### 6.1 完整的 HybridMemoryStore 实现

```python
# nanobot/agent/memory.py

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class MemoryChunk:
    """检索到的记忆片段"""
    path: str
    title: str
    content: str
    score: float


class HybridMemoryStore:
    """
    混合记忆存储：原生文件 + QMD 语义检索

    - 保留原有 MEMORY.md 和 YYYY-MM-DD.md 结构
    - 可选启用 qmd 进行语义搜索
    - 自动降级到纯文本模式
    """

    def __init__(self, workspace: Path, enable_qmd: bool = True):
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"

        self._qmd_enabled = enable_qmd and self._check_qmd()
        self._initialized = False

    def _check_qmd(self) -> bool:
        """检查 qmd 是否可用"""
        return shutil.which("qmd") is not None

    async def initialize(self) -> None:
        """初始化 qmd 索引"""
        if self._initialized or not self._qmd_enabled:
            return

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 检查是否已有 memory 集合
            result = await self._run_qmd("collection", "list", "--json")
            collections = json.loads(result)

            has_memory = any(c["name"] == "memory" for c in collections)

            if not has_memory:
                # 创建集合
                await self._run_qmd(
                    "collection", "add", str(self.memory_dir),
                    "--name", "memory"
                )

                # 添加上下文
                await self._run_qmd(
                    "context", "add", "qmd://memory",
                    "Agent memory: long-term facts and daily notes"
                )

                # 初始嵌入
                await self._run_qmd("embed")

            self._initialized = True
            logger.info("QMD memory index initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize QMD: {e}")
            self._qmd_enabled = False

    async def _run_qmd(self, *args: str) -> str:
        """执行 qmd 命令"""
        proc = await asyncio.create_subprocess_exec(
            "qmd", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"qmd {args[0]} failed: {stderr.decode()}")

        return stdout.decode()

    async def search(self, query: str, limit: int = 5) -> list[MemoryChunk]:
        """
        语义搜索记忆

        Args:
            query: 搜索查询
            limit: 返回结果数量

        Returns:
            相关记忆片段列表
        """
        if not self._qmd_enabled:
            return self._fallback_search(query, limit)

        try:
            result = await self._run_qmd(
                "query", query,
                "-n", str(limit),
                "--min-score", "0.2",
                "-c", "memory",
                "--json"
            )

            docs = json.loads(result)

            return [
                MemoryChunk(
                    path=doc["path"],
                    title=doc.get("title", Path(doc["path"]).stem),
                    content=doc.get("snippet", ""),
                    score=doc["score"]
                )
                for doc in docs
            ]

        except Exception as e:
            logger.warning(f"QMD search failed: {e}")
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> list[MemoryChunk]:
        """降级搜索：简单关键词匹配"""
        keywords = set(query.lower().split())
        results = []

        # 搜索所有 markdown 文件
        for md_file in self.memory_dir.glob("**/*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                content_lower = content.lower()

                # 计算匹配分数
                matches = sum(1 for k in keywords if k in content_lower)
                if matches > 0:
                    score = matches / len(keywords)
                    results.append(MemoryChunk(
                        path=str(md_file.relative_to(self.memory_dir)),
                        title=md_file.stem,
                        content=content[:500] + ("..." if len(content) > 500 else ""),
                        score=score
                    ))
            except Exception:
                continue

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def get_context_for_query(
        self,
        query: Optional[str] = None,
        include_long_term: bool = True,
        include_today: bool = True
    ) -> str:
        """
        获取与查询相关的记忆上下文

        Args:
            query: 用户查询（用于语义检索）
            include_long_term: 是否包含长期记忆
            include_today: 是否包含今日笔记

        Returns:
            格式化的记忆上下文
        """
        parts = []

        # 如果有查询，使用语义检索
        if query and self._qmd_enabled:
            chunks = await self.search(query, limit=5)
            if chunks:
                memories = []
                for chunk in chunks:
                    memories.append(
                        f"### {chunk.title} (relevance: {chunk.score:.0%})\n{chunk.content}"
                    )
                parts.append("## Relevant Memories\n\n" + "\n\n".join(memories))

        # 长期记忆（MEMORY.md）
        if include_long_term and self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")
            if content.strip():
                parts.append(f"## Long-term Memory\n\n{content}")

        # 今日笔记
        if include_today:
            today_file = self._get_today_file()
            if today_file.exists():
                content = today_file.read_text(encoding="utf-8")
                if content.strip():
                    parts.append(f"## Today's Notes\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    # === 原有方法保留兼容 ===

    def _get_today_file(self) -> Path:
        from datetime import datetime
        return self.memory_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    def read_today(self) -> str:
        today_file = self._get_today_file()
        return today_file.read_text(encoding="utf-8") if today_file.exists() else ""

    async def append_today(self, content: str) -> None:
        """追加今日笔记并更新索引"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        today_file = self._get_today_file()

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            content = existing + "\n" + content
        else:
            from datetime import datetime
            header = f"# {datetime.now().strftime('%Y-%m-%d')}\n\n"
            content = header + content

        today_file.write_text(content, encoding="utf-8")

        # 更新 qmd 索引
        if self._qmd_enabled:
            try:
                await self._run_qmd("update")
            except Exception as e:
                logger.warning(f"Failed to update QMD index: {e}")

    def read_long_term(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    async def write_long_term(self, content: str) -> None:
        """写入长期记忆并更新索引"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(content, encoding="utf-8")

        if self._qmd_enabled:
            try:
                await self._run_qmd("update")
            except Exception as e:
                logger.warning(f"Failed to update QMD index: {e}")
```

### 6.2 在 ContextBuilder 中使用

```python
# nanobot/agent/context.py (修改)

class ContextBuilder:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = HybridMemoryStore(workspace)  # 替换原来的 MemoryStore
        self.skills = SkillsLoader(workspace)

    async def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        user_query: str | None = None  # 新增：用于语义检索
    ) -> str:
        parts = []

        # ... 其他代码保持不变 ...

        # Memory context - 现在支持语义检索
        await self.memory.initialize()
        memory = await self.memory.get_context_for_query(query=user_query)
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # ... 其他代码保持不变 ...
```

---

## 七、配置选项

```yaml
# config.yaml
memory:
  enable_qmd: true          # 是否启用 qmd
  qmd_daemon: true          # 是否使用 daemon 模式
  qmd_http_port: 8181       # HTTP 端口
  search_limit: 5           # 默认返回结果数
  min_score: 0.2            # 最低相关性阈值
  include_long_term: true   # 是否包含长期记忆
  include_today: true       # 是否包含今日笔记
```

---

## 八、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| qmd 项目不再维护 | 低 | 高 | 锁定版本，必要时 fork |
| 模型下载失败 | 中 | 中 | 提供镜像或预打包 |
| Windows 兼容性 | 中 | 中 | 测试并记录限制 |
| GPU 不可用 | 中 | 低 | CPU fallback 自动启用 |
| 索引损坏 | 低 | 中 | 提供重建命令 |

---

## 九、结论

### 推荐：采用 QMD 集成

**理由:**

1. **需求匹配度高:** qmd 专门为 AI agent 记忆场景设计
2. **技术风险可控:** 渐进式集成，完善的降级策略
3. **收益明显:** 语义检索能力将显著提升 nanobot 的记忆效果
4. **生态协同:** MCP 协议与 nanobot 的扩展方向一致

### 实施建议

1. **第一阶段:** CLI 调用方式快速验证
2. **第二阶段:** 升级为 MCP HTTP 模式
3. **第三阶段:** 根据使用反馈优化检索策略

---

*Document Status: Ready for Review*
*Next Step: Create the plan*

---

# 附录：Claude-Mem 对比分析

## Claude-Mem 概述

[github.com/thedotmack/claude-mem](https://github.com/thedotmack/claude-mem)

**定位:** 专为 Claude Code 设计的持久化记忆压缩系统

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude-Mem 架构                          │
├─────────────────────────────────────────────────────────────┤
│  Lifecycle Hooks (5个)                                      │
│  ├── SessionStart      → 加载历史上下文                    │
│  ├── UserPromptSubmit  → 捕获用户输入                      │
│  ├── PostToolUse       → 记录工具使用观察                  │
│  ├── Stop              → 处理停止事件                      │
│  └── SessionEnd        → 生成会话摘要                      │
├─────────────────────────────────────────────────────────────┤
│  存储层                                                     │
│  ├── SQLite + FTS5     → 全文搜索                          │
│  └── Chroma Vector DB  → 语义搜索                          │
├─────────────────────────────────────────────────────────────┤
│  Worker Service                                             │
│  ├── HTTP API (port 37777)                                 │
│  ├── Web Viewer UI                                          │
│  └── 10 个搜索端点                                          │
├─────────────────────────────────────────────────────────────┤
│  MCP 工具 (5个)                                             │
│  ├── search            → 全文索引搜索                      │
│  ├── timeline          → 时间线上下文                      │
│  ├── get_observations  → 获取完整详情                      │
│  ├── save_memory       → 手动保存记忆                      │
│  └── __IMPORTANT       → 工作流文档                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 三方对比：Nanobot 现有 vs QMD vs Claude-Mem

| 维度 | Nanobot 现有 | QMD | Claude-Mem |
|------|-------------|-----|------------|
| **定位** | 简单文件存储 | 文档搜索引擎 | Claude 会话记忆 |
| **目标用户** | nanobot | 任何 AI agent | Claude Code |
| **存储格式** | Markdown 文件 | Markdown 文件 | SQLite 数据库 |
| **全文搜索** | ❌ 无 | ✅ FTS5 (BM25) | ✅ FTS5 |
| **向量搜索** | ❌ 无 | ✅ 本地模型 | ✅ Chroma |
| **重排序** | ❌ 无 | ✅ 本地 LLM | ❌ 无 |
| **自动捕获** | ❌ 手动 | ❌ 手动索引 | ✅ Hooks 自动 |
| **会话感知** | ❌ 无 | ❌ 无 | ✅ 会话/观察 |
| **摘要生成** | ❌ 无 | ❌ 无 | ✅ LLM 摘要 |
| **隐私控制** | ✅ 本地 | ✅ 完全本地 | ⚠️ 需 LLM API |
| **运行时** | Python | Bun + GGUF | Node.js + uv |
| **模型依赖** | 无 | ~2GB 本地 | Chroma embeddings |
| **MCP 支持** | ❌ 无 | ✅ 原生 | ✅ 原生 |
| **开源协议** | MIT | MIT | AGPL-3.0 |

---

## Claude-Mem 的独特价值

### 1. 自动观察捕获

```typescript
// PostToolUse Hook - 自动记录工具使用
{
  type: "file_edit",
  title: "Fixed login validation bug",
  content: "Modified auth.py lines 45-67...",
  semantic_summary: "Bug fix for password validation"
}
```

### 2. 渐进式披露（Progressive Disclosure）

```
3-Layer Workflow（节省 ~10x tokens）：

1. search()        → 返回紧凑索引 (~50-100 tokens/结果)
2. timeline()      → 返回时间上下文
3. get_observations() → 仅获取需要的完整详情 (~500-1000 tokens/结果)
```

### 3. 会话连续性

```
Session A (结束)
    ↓ 自动生成摘要
    ↓ 存储到 SQLite + Chroma
Session B (开始)
    ↓ SessionStart Hook
    ↓ 注入相关上下文
    → Claude 记得之前的工作
```

---

## Claude-Mem 的局限性

| 局限 | 影响 | 对 nanobot 的意义 |
|------|------|------------------|
| **专为 Claude Code 设计** | Hooks 机制不适用 | nanobot 需要适配 |
| **AGPL-3.0 协议** | 修改需开源 | 商业使用需考虑 |
| **需要 LLM API** | 摘要生成需调用 LLM | 增加成本 |
| **Node.js 生态** | 与 Python 集成复杂 | 需要桥接层 |
| **复杂度高** | 6个 hooks + worker | 维护成本高 |

---

## 最佳方案：混合架构

基于分析，我推荐**分阶段采用 QMD + 借鉴 Claude-Mem 理念**：

### 架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                      Nanobot 智能记忆系统                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │   写入层     │    │   存储层     │    │   检索层     │        │
│   ├──────────────┤    ├──────────────┤    ├──────────────┤        │
│   │ 手动写入     │    │ Markdown    │    │ QMD          │        │
│   │ (现有逻辑)   │───►│ 文件        │◄───│ 混合搜索     │        │
│   │              │    │              │    │              │        │
│   │ 自动观察     │    │ workspace/   │    │ BM25+向量    │        │
│   │ (新增)       │───►│ memory/      │    │ +重排序      │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│         ▲                                        │                 │
│         │                                        │                 │
│   ┌─────┴────────────────────────────────────────┴─────┐          │
│   │                  Nanobot Agent Loop                 │          │
│   ├─────────────────────────────────────────────────────┤          │
│   │  1. 工具调用后 → 自动记录观察                       │          │
│   │  2. 对话结束后 → 生成会话摘要                       │          │
│   │  3. 构建上下文时 → 语义检索相关记忆                 │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 分阶段实施

#### Phase 1: QMD 基础集成（已规划）
- 用 QMD 替换简单的 `get_recent_memories()`
- 实现语义检索
- 降级策略保障稳定性

#### Phase 2: 借鉴 Claude-Mem 的观察系统

```python
# nanobot/agent/observation.py

@dataclass
class Observation:
    """工具使用观察记录"""
    id: str
    timestamp: datetime
    type: str  # file_edit, shell_exec, web_search, etc.
    title: str
    content: str
    metadata: dict

class ObservationRecorder:
    """在 AgentLoop 中自动记录工具调用"""

    async def record(self, tool_name: str, args: dict, result: Any) -> None:
        """记录工具使用观察"""
        obs = Observation(
            id=generate_id(),
            timestamp=datetime.now(),
            type=self._classify_tool(tool_name),
            title=self._generate_title(tool_name, args, result),
            content=self._format_content(tool_name, args, result),
            metadata={"tool": tool_name, "args": args}
        )

        # 写入 Markdown 文件（供 QMD 索引）
        await self._write_observation(obs)
```

#### Phase 3: 渐进式披露检索

```python
# nanobot/agent/memory.py

class SmartMemoryRetrieval:
    """三层渐进式检索，节省 tokens"""

    async def get_context(self, query: str) -> str:
        # Layer 1: 获取索引（紧凑）
        index = await self.qmd.search(query, limit=20, format="index")

        # Layer 2: 评估相关性
        relevant_ids = [r.id for r in index if r.score > 0.4]

        # Layer 3: 仅获取高相关内容的详情
        if len(relevant_ids) <= 5:
            details = await self.qmd.get_details(relevant_ids)
        else:
            # 太多结果，只用摘要
            details = [r.snippet for r in index[:5]]

        return self._format_context(details)
```

#### Phase 4: 会话摘要（可选）

```python
# nanobot/agent/summarizer.py

class SessionSummarizer:
    """会话结束时生成摘要"""

    async def summarize_session(self, session_id: str) -> str:
        # 获取会话中的所有观察
        observations = await self.memory.get_session_observations(session_id)

        # 使用 LLM 生成摘要
        prompt = f"Summarize these observations:\n{observations}"
        summary = await self.provider.generate(prompt)

        # 存储为长期记忆
        await self.memory.write_long_term(
            f"## Session {session_id}\n{summary}"
        )
```

---

## 最终推荐

| 方案 | 推荐度 | 理由 |
|------|--------|------|
| **QMD 集成** | ⭐⭐⭐⭐⭐ | 完全本地、MIT 协议、混合检索最强 |
| **Claude-Mem 理念借鉴** | ⭐⭐⭐⭐ | 观察系统和渐进披露值得学习 |
| **Claude-Mem 直接集成** | ⭐⭐ | AGPL 协议、Claude Code 专用、复杂度高 |

### 行动计划

1. **立即执行：** Phase 1 - QMD 基础集成
2. **下一步：** Phase 2 - 实现观察记录系统
3. **评估后：** Phase 3/4 - 渐进披露 + 会话摘要

---

*附录更新: 2026-02-11*
*包含 Claude-Mem 对比分析*
