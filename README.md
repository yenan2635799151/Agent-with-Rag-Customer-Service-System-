# 🤖 Agent with RAG Customer Service System

基于 LangChain ReAct Agent 和 RAG 技术的智能客服系统，专为产品售后服务设计。



## ✨ 功能特性

- **🧠 智能对话**：基于 ReAct 架构的 AI Agent，支持多轮对话和工具调用
- **📚 RAG 检索增强**：混合检索策略（Dense + BM25 + Cross-encoder 重排序）
- **🌐 实时信息**：集成天气查询、位置服务等外部 API
- **📊 动态提示词**：根据对话上下文自动切换系统提示词
- **⚡ 流式响应**：支持 SSE 流式传输，实时返回 AI 回复
- **🎯 效果评估**：内置 RAGAS 评估框架，量化系统性能

## 🏗️ 技术架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   FastAPI       │────▶│   ReAct Agent   │
│   Frontend      │◀────│   Backend       │◀────│   (LangChain)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌─────────────┐           ┌─────────────┐
                        │  Milvus DB  │           │  7 Tools    │
                        │  (Primary)  │           │  + Weather  │
                        └─────────────┘           │  + Location │
                        ┌─────────────┐           │  + RAG etc. │
                        │ Chroma DB   │           └─────────────┘
                        │ (Fallback)  │
                        └─────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- 可选：Docker（用于 Milvus 向量数据库）

### 1. 克隆项目

```bash
git clone https://github.com/yenan2635799151/Agent-with-Rag-Customer-Service-System-.git
cd Agent-with-Rag-Customer-Service-System-
```

### 2. 后端配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥（编辑 config/ 目录下的 yaml 文件）
# - config/rag.yml: DeepSeek API Key
# - config/agent.yml: 高德地图 API Key
```

### 3. 初始化向量数据库

**方式1：使用 Milvus（推荐，需要 Docker）**

```bash
# 1. 启动 Milvus Docker 容器
bash milvus.sh

# 2. 加载知识库文档到 Milvus
python -c "from rag.m_vector_store import MilvusVectorStoreService; vs = MilvusVectorStoreService(); vs.load_documents()"
```

**方式2：使用 Chroma（无需 Docker，快速开始）**

```bash
# 加载知识库文档到 Chroma
python -c "from rag.vector_store import VectorStoreService; vs = VectorStoreService(); vs.load_documents()"
```

### 4. 启动后端服务

```bash
# 方式1: 直接运行
python api.py

# 方式2: 使用 uvicorn（推荐开发）
uvicorn api:app --reload --port 8004
```

### 5. 前端配置

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000 即可使用系统。

## 📁 项目结构

```
.
├── agent/                      # Agent 核心逻辑
│   ├── react_agent.py         # ReAct Agent 初始化
│   ├── tools/                 # 工具定义
│   │   ├── agent_tools.py     # 7个工具实现
│   │   └── middleware.py      # 工具中间件
│   └── memory_test.py         # 记忆测试
├── api.py                     # FastAPI 入口
├── app.py                     # Streamlit 备用 UI
├── config/                    # 配置文件
│   ├── rag.yml               # RAG 配置（API Key、模型参数）
│   ├── chroma.yml            # Chroma 数据库配置
│   ├── agent.yml             # Agent 配置（外部 API Key）
│   └── prompts.yml           # 提示词文件路径
├── data/                      # 知识库文档（PDF/TXT）
├── eval/                      # 评估模块
│   └── eval_pipline.py       # RAGAS 评估脚本
├── frontend/                  # Next.js 前端
│   ├── src/
│   │   ├── app/              # 页面路由
│   │   ├── components/       # React 组件
│   │   └── hooks/            # 自定义 Hooks
│   └── next.config.ts        # Next.js 配置
├── model/                     # 模型工厂
│   └── factory.py            # LLM & Embedding 初始化
├── prompts/                   # 提示词模板
│   ├── main_prompt.txt       # 主系统提示词
│   └── report_prompt.txt     # 报告生成提示词
├── rag/                       # RAG 模块
│   ├── m_vector_store.py     # Milvus 向量存储（首选）
│   ├── vector_store.py       # Chroma 向量存储（备选）
│   ├── rag_service.py        # RAG 服务封装
│   └── rag_tools/            # RAG 工具
│       ├── bm25.py           # BM25 检索
│       ├── hybrid.py         # 混合检索器
│       ├── CE_reranker.py    # Cross-encoder 重排序
│       └── rrf.py            # RRF 融合算法
├── utils/                     # 工具函数
│   ├── config_handler.py     # 配置管理
│   ├── logger_handler.py     # 日志处理
│   └── prompt_loader.py      # 提示词加载
└── test.py                    # 测试脚本
```

## ⚙️ 配置说明

所有配置位于 `config/` 目录下，无需 `.env` 文件：

| 文件 | 配置项 | 说明 |
|------|--------|------|
| `rag.yml` | `api_key` | DeepSeek API Key |
| | `base_url` | DeepSeek API 地址 |
| | `embed_model_path` | 本地 Embedding 模型路径 |
| | `vector_k`, `bm25_k`, `final_k` | 检索参数 |
| `chroma.yml` | `collection_name` | Chroma 集合名称 |
| | `persist_dir` | 持久化目录（默认：`choma_db/`） |
| | `chunk_size`, `chunk_overlap` | 文档分块参数 |
| `agent.yml` | `amap_key` | 高德地图 API Key |
| | `external_csv_path` | 外部数据 CSV 路径 |
| `milvus.yml` | `host` | Milvus 服务器地址（默认：localhost） |
| | `port` | Milvus 端口（默认：19530） |
| | `collection_name` | Milvus 集合名称 |

## 🔌 API 接口

### 主要端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat` | 聊天接口（SSE 流式） |

### 请求示例

```bash
curl -X POST http://localhost:8004/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请问这款吸尘器怎么清洗？",
    "session_id": "user_123"
  }'
```

### 响应格式

SSE 流式响应，每行一个 JSON 对象：

```json
{"type": "token", "content": "您"}
{"type": "token", "content": "好"}
{"type": "token", "content": "！"}
{"type": "tool", "name": "rag_summarize", "input": "吸尘器清洗方法"}
{"type": "tool_result", "name": "rag_summarize", "output": "..."}
{"type": "end"}
```

## 🛠️ 开发指南

### 测试 Agent

```bash
# 直接测试 Agent
python agent/react_agent.py
```

### 测试向量检索

```bash
# 测试 Milvus 检索效果（首选）
python rag/m_vector_store.py

# 或测试 Chroma 检索效果（备选）
python rag/vector_store.py
```

### 运行评估

```bash
# RAGAS 评估（默认使用 Milvus）
python eval/eval_pipline.py
```

### 切换向量数据库

```python
# 使用 Milvus（推荐，需要 Docker）
from rag.m_vector_store import MilvusVectorStoreService
vs = MilvusVectorStoreService()

# 使用 Chroma（无需 Docker）
from rag.vector_store import VectorStoreService
vs = VectorStoreService()
```

## 🧩 核心组件

### RAG Pipeline

混合检索流程（支持 Milvus/Chroma）：

1. **Dense Retrieval**: Milvus/Chroma 向量检索（top 30）
2. **BM25 Reranking**: 稀疏向量重排序（top 10）
3. **Cross-encoder**: 精排模型（top 3，模型不可用时跳过）

### 工具列表

| 工具名 | 功能 |
|--------|------|
| `rag_summarize` | RAG 知识检索与总结 |
| `get_weather` | 获取实时天气信息 |
| `get_user_location` | 获取用户地理位置 |
| `get_user_id` | 获取用户 ID |
| `get_current_month` | 获取当前月份 |
| `fetch_external_data` | 获取外部 CSV 数据 |
| `fill_context_for_report` | 触发报告生成模式 |

### 动态提示词

通过 `fill_context_for_report` 工具触发，自动切换为报告生成专用提示词。


## 📊 性能评估

系统内置 RAGAS 评估框架，支持以下指标：

- **Faithfulness**: 回答与检索内容的忠实度
- **Answer Relevancy**: 回答相关性
- **Context Precision**: 上下文精确率
- **Context Recall**: 上下文召回率





## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [Milvus](https://milvus.io/) - 云原生向量数据库
- [Chroma](https://www.trychroma.com/) - 轻量级向量数据库
- [DeepSeek](https://www.deepseek.com/) - 大语言模型

---

<p align="center">
  Made with ❤️ for better customer service
</p>
