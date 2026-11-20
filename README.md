# 文档 RAG 问答系统（BM25 / Elasticsearch）

面向小规模文档知识库的 RAG 问答原型：先把文档切成段落并建立 BM25 索引，再把检索到的原文片段连同来源和得分一起注入提示词，让大模型只依据证据回答。项目同时保留一个 Elasticsearch 分支，用于验证文档规模增长后的扩展路径。

## 演示视频

[点击观看项目演示视频](https://github.com/kk-20040526/rag-doc-qa/releases/tag/v1.0.0)

视频覆盖 BM25 正常检索、无命中拒答，以及 Elasticsearch 大文件检索示例。

## 运行截图

BM25 检索答案与来源展示：

![BM25 检索答案与来源展示](images/bm25_streamlit.png)

无命中时的拒答：

![无命中时的拒答](images/rag_no_hit.png)

Elasticsearch 分支的检索结果：

![Elasticsearch 分支的检索结果](images/es_streamlit.png)

## 项目定位

项目重点不是“换更大的模型”，而是处理 RAG 里的三个产品问题：

1. 用户的问题能不能找到对应资料；
2. 回答能不能展示来源，让用户核对依据；
3. 找不到依据时，系统是否敢明确说不知道。

## 两套检索方案的分工

- **BM25 内存检索**：不依赖外部服务，适合小规模文档；当前 14 篇、86 段，启动成本低，适合作为默认方案。
- **Elasticsearch 分支**：面向文档规模增长后的扩展方案，索引和检索从进程内移到独立服务；更适合后续做多文档、增量索引和水平扩展。
- 当前 ES 为本地单节点验证，用于确认扩展路径可行。

## 当前实现

- 文档按空行切分为段落；
- 使用 jieba 做中文分词，过滤停用词和纯标点，保留有意义的单字；
- 使用 rank_bm25 的 BM25Okapi 建立内存索引；
- 检索 Top-5 片段，并对命中片段补充同源上一段和下一段；
- 将检索证据直接注入提示词，由 qwen-flash 组织答案；
- 界面展示命中片段、来源文件和 BM25 得分；
- 没有检索到内容时，提示“知识库中没有相关内容”；
- 侧边栏可切换 BM25 / Elasticsearch 两套方案，ES 模式支持从同一批 86 段构建索引并返回 Top-5。

当前知识库包含 14 篇文档、86 个段落，平均每段约 133.5 字。

## 技术栈

| 模块 | 技术 |
|---|---|
| 界面 | Streamlit |
| 检索 | rank_bm25 + jieba |
| 大模型 | 阿里云百炼 qwen-flash |
| Agent 框架 | qwen-agent（仅用于对话封装） |
| Elasticsearch 分支 | 本地 Docker 8.11 已验证，非生产集群 |
| 语言 | Python 3.11 |

## 处理链路

```text
本地 txt 文档
    ↓ 按空行切分
86 个段落
    ↓ jieba 分词、停用词过滤
BM25 内存索引
    ↓ 用户提问
Top-5 段落 + 同源相邻片段
    ↓ 证据注入提示词
qwen-flash 生成答案
    ↓
展示答案、来源文件、BM25 得分
```

ES 分支的链路：

```text
86 个段落
    ↓ bulk 写入 doc_index
Elasticsearch 单节点
    ↓ match 查询 content 字段
Top-5 命中片段
    ↓ 取前 3 段作为上下文
qwen-flash 生成答案
    ↓
展示片段、来源、ES 相关性得分
```

## 已验证结果

核实日期：2026-11-17。

| 项目 | 结果 |
|---|---|
| 文档 / 段落 | 14 篇 / 86 段 |
| 索引构建 | 冷启动约 0.6—0.9 秒，含 jieba 首次加载；预热后约 30—50 毫秒 |
| 单次检索 | 完整工具调用约 0.45—1.7 毫秒，纯 BM25 打分约 0.04—0.06 毫秒 |
| 域内问题 | 4 个预设问题 Top-1 均命中 |
| 域外问题 | 无命中时提示“知识库中没有相关内容”；弱相关召回时由模型判断证据不足并拒答 |
| ES 本地验证 | 86 段写入 `doc_index`；同一问题 Top-1 为 `02_BM25检索算法详解.txt`，得分 21.22 |

4 个预设域内问题：

| 问题 | Top-1 来源 | BM25 得分 |
|---|---|---|
| BM25 的 k1 和 b 参数分别控制什么 | `02_BM25检索算法详解.txt` | 14.70 |
| PaddleOCR 识别模块用什么算法 | `12_自然场景OCR系统项目概览.txt` | 6.73 |
| Elasticsearch 索引建好了为什么搜不到 | `13_工程实践常见问题与排查.txt` | 13.23 |
| RAG 为什么缓解幻觉 | `06_大模型幻觉与RAG的边界.txt` | 8.80 |

这 4 个问题为预设演示问题，非独立测试集，不代表整体检索准确率。

## 快速开始

### 1. 创建环境

```powershell
cd 02_rag_doc_qa
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 配置 Key

复制 `.env.example` 为 `.env`，填入自己的百炼 Key：

```text
DASHSCOPE_API_KEY=sk-你的百炼Key
QWEN_MODEL=qwen-flash
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`.env` 已被 `.gitignore` 排除，不要提交。

### 3. 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run final_app.py
```

浏览器打开 `http://localhost:8501`，默认进入 BM25 内存检索模式。

### 4. 可选：启动本地 Elasticsearch 分支

ES 分支需要一个本地 ES 服务。Windows 上最省事的方式是用 Docker Desktop：

```powershell
docker run -d --name rag-es -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" `
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

启动后浏览器访问 `http://127.0.0.1:9200`，能看到 `"tagline": "You Know, for Search"` 即为正常。

ES 分支会递归读取 `docs/` 目录下的 `.txt` / `.md` 文件；大文本测试可以放到 `docs/large/`。当前实现会先保留原有空行段落，超过 700 字的段落再按固定长度滑动切块，重叠 120 字。

然后在 Streamlit 左侧切换到“海量文档方案（Elasticsearch）”，侧边栏会显示读取目录、文件数和片段数；点“构建ES索引”，成功后即可用同一批文档检索。

说明：

- 代码里 ES 地址固定为 `http://127.0.0.1:9200`。Windows 上 `localhost` 可能先解析到 IPv6，导致连接超时。
- 这里关闭了 `xpack.security`，只用于本地演示，不是生产配置。
- 这是单节点、86 个段落的本地验证，不代表生产级集群或海量数据性能。

## 演示问题

1. BM25 的 k1 和 b 参数分别控制什么？
2. PaddleOCR 识别模块用什么算法？
3. Elasticsearch 索引建好了为什么搜不到？
4. RAG 为什么缓解幻觉？
5. 今天火星的天气怎么样？

前四个用于展示“答案 + 来源 + 得分”，第五个用于展示无命中时的拒答。

## 目录结构

```text
02_rag_doc_qa/
├── final_app.py          # Streamlit 主程序
├── kb_tool.py            # BM25KBTool：jieba 分词、索引、检索、邻接扩展
├── es_retrieval.py       # Elasticsearch 检索分支：建索引、检索、删除索引
├── docs/                 # 14 篇演示文档，共 86 个段落
├── images/               # 界面与旧版演示截图
├── requirements.txt
├── .env.example
└── README.md
```

## 边界与不足

- 这是小规模原型，知识库只有 14 篇文档、86 个段落。
- 当前检索层没有分数阈值，只过滤 `score > 0`，弱相关问题可能召回无关片段。
- 拒答依赖提示词和模型判断，不是确定性规则。
- BM25 对同义表达和多跳问题能力有限。
- 当前是内存索引，文档更新、并发和权限控制都未做企业级处理。
- Elasticsearch 分支为本地单节点验证，关闭了安全认证，未做集群、权限、监控与性能压测。
- 当前段落规模较小，ES 分支的价值在于验证扩展路径的可行性。
- 当前主链路是应用直接检索后注入提示词，不是模型通过 Function Calling 自动选择工具。

## 后续计划

1. 扩充域内、域外和相似文档干扰的评测问题；
2. 统计 Recall@K、MRR，并校准命中阈值；
3. 比较 BM25 与向量检索的混合召回效果；
4. 增加来源引用校验和失败案例分析。
