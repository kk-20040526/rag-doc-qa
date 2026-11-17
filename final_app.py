# -*- coding: utf-8 -*-
"""
AI 搜索类应用（综合版）
整合 BM25 内存检索与 Elasticsearch 检索两种方案，
基于 Streamlit 构建可视化交互界面，结合大模型实现 RAG 问答。

功能：
- 小文档方案：自研 BM25 检索 + 证据注入问答
- 海量文档方案：Elasticsearch 全文检索 + 大模型生成答案
- 侧边栏模式切换、ES 索引管理
"""
import os
import streamlit as st
from dotenv import load_dotenv
from es_retrieval import ESRetrieval
from qwen_agent.llm import get_chat_model
from qwen_agent.agents import Assistant
# 说明：qwen-agent 的 Retrieval 只接受 {"query", "files"}（现场解析文件），
# 不支持 Retrieval({"docs": docs, "method": "bm25"}) 这种塞段落的写法，
# 旧写法会走 files=[] 分支返回空检索。改用本地 BM25 工具（见 kb_tool.py）。
from kb_tool import BM25KBTool

# 读取同目录下的 .env。
# 原代码只用 os.getenv 取值，却没有任何一处加载 .env —— 把 Key 写进 .env 也不会生效。
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI搜索-综合版",
    page_icon="🔍",
    layout="wide"
)

# ==================== LLM 配置 ====================
# 三个环境变量：
#   QWEN_MODEL     模型名，例如 qwen-turbo（百炼）或 deepseek-chat（DeepSeek）
#   QWEN_BASE_URL  OpenAI 兼容的 base url
#   LLM_API_KEY / DASHSCOPE_API_KEY  密钥（LLM_API_KEY 优先，换供应商时不用改名）
#
# 注意键名必须是 model_server：qwen-agent 只认这个键。写成 base_url 会被忽略，
# 然后按模型名猜服务商（含 qwen 就走 DashScope SDK），换 DeepSeek 一定失败。
llm_cfg = {
    'model': os.getenv('QWEN_MODEL', 'qwen-turbo'),
    'api_key': os.getenv('LLM_API_KEY') or os.getenv('DASHSCOPE_API_KEY', ''),
    'model_server': os.getenv(
        'QWEN_BASE_URL',
        'https://dashscope.aliyuncs.com/compatible-mode/v1'
    ),
}

# ==================== 侧边栏：检索模式切换 ====================
st.sidebar.header("⚙️ 搜索模式")
mode = st.sidebar.radio(
    "选择检索方案：",
    ["📚 小文档方案（BM25内存检索）", "🗄️ 海量文档方案（Elasticsearch）"],
    index=0,   # 默认落在 BM25：本机没有 ES 服务，默认进 ES 模式会一问就报连不上
)

# ==================== 工具函数：加载文档 ====================
def load_documents(docs_dir="./docs"):
    """读取 docs 目录下所有 txt 文档，按段落切分"""
    docs = []
    if not os.path.exists(docs_dir):
        return docs
    for f in os.listdir(docs_dir):
        if f.endswith(".txt"):
            with open(os.path.join(docs_dir, f), "r", encoding="utf-8") as file:
                text = file.read()
            for para in text.split("\n\n"):
                if para.strip():
                    docs.append({"content": para.strip(), "source": f})
    return docs


# ==================== 缓存：内存知识库 ====================
@st.cache_resource
def get_kb():
    """构建 BM25 内存知识库（小文档方案）"""
    docs = load_documents()
    return BM25KBTool({"docs": docs})


# ==================== 缓存：ES 连接 ====================
@st.cache_resource
def get_es():
    """获取 ES 检索实例"""
    return ESRetrieval()


# ==================== 侧边栏：ES 知识库管理 ====================
with st.sidebar:
    if "海量文档" in mode:
        st.header("🗄️ ES 知识库管理")

        if st.button("🔄 构建ES索引"):
            with st.spinner("正在构建索引..."):
                chunks = load_documents()
                es = get_es()
                es.delete_index()  # 先清旧索引
                es.add_documents(chunks)
                st.success(f"✅ 已索引 {len(chunks)} 个片段！")

        if st.button("🗑️ 删除ES索引"):
            es = get_es()
            es.delete_index()
            st.success("索引已删除")

    st.markdown("---")
    st.caption("💡 小文档方案无需数据库，适合 1000 段以内")
    st.caption("💡 ES 方案支持水平扩展，适合海量文档")


# ==================== 主区域：搜索交互 ====================
st.title("🔍 AI 文档智能搜索问答")
st.markdown("基于 **RAG（检索增强生成）** 技术，结合本地文档知识库与大语言模型，精准回答文档相关问题。")

query = st.text_input(
    "输入你的问题",
    placeholder="例如：什么是大语言模型？"
)

if st.button("🚀 搜索", type="primary") and query:
    with st.spinner("⚡ 正在检索并生成答案..."):

        if "小文档" in mode:
            # ---------- 方案一：BM25 检索 + 片段注入生成 ----------
            kb = get_kb()
            evidence = kb.call({"query": query})   # 先检索，把"依据"露给用户看
            # 注意：这里不再依赖模型自己二次调工具取片段。实测模型会把问题改写成
            # 别的词再检索，命中的片段可能恰好不含答案那句话，于是答成"文档里没写"。
            # 改成把本次检索结果直接注入上下文（经典 RAG 做法），行为可复现。
            prompt = (
                "以下是知识库中检索到的片段：\n\n"
                f"{evidence}\n\n"
                f"问题：{query}\n"
                "请只依据上面的片段回答，引用时标出片段号；"
                "片段里确实没有的内容就说「知识库中没有相关内容」，不要用常识补充。"
            )
            bot = Assistant(
                llm=llm_cfg,
                system_message="你是知识库问答助手，只能依据用户给出的检索片段回答，不得编造。",
            )
            response = bot.run([{'role': 'user', 'content': prompt}])
            # qwen-agent 0.0.34 的 run() 每轮产出的是「整段消息列表」，
            # 不是单个 dict —— 旧写法 list(response)[-1].get(...) 会抛
            # AttributeError: 'list' object has no attribute 'get'
            turns = list(response)
            last = turns[-1] if turns else []
            if isinstance(last, list):
                last = last[-1] if last else {}
            answer = last.get("content", "生成失败") if isinstance(last, dict) else str(last)
            if not answer or not str(answer).strip():
                answer = "生成失败（模型没有返回内容）"

            st.subheader("🤖 AI综合回答")
            st.success(answer)

            with st.expander("📚 检索到的文档片段（BM25，回答的依据）"):
                st.text(evidence)

        else:
            # ---------- 方案二：ES 检索 + 大模型生成 ----------
            es = get_es()
            results = es.search(query, top_k=5)

            # 展示检索结果
            st.subheader("📋 ES检索结果（BM25排序）")
            if results:
                for idx, r in enumerate(results, 1):
                    with st.expander(
                        f"片段 {idx}  |  来源：{r['source']}  |  得分：{r['score']:.2f}",
                        expanded=(idx <= 2)
                    ):
                        st.write(r['content'])
            else:
                st.info("未检索到相关文档片段")

            # 拼接 Top3 片段作为上下文，调用大模型生成答案
            if results:
                context = "\n\n".join([r["content"] for r in results[:3]])
                prompt = f"""请根据以下文档内容回答问题。
如果文档中没有相关信息，请如实回答"文档中未找到相关内容"，不要编造。

文档内容：
{context}

问题：{query}"""

                llm = get_chat_model(llm_cfg)
                response = llm.chat([{"role": "user", "content": prompt}])
                response_list = list(response)

                # qwen-agent 的 chat() 每轮可能返回「消息列表」，不是单个 dict。
                # 这里统一拆到最后一层，再取 content，避免页面直接打印原始列表。
                last = response_list[-1] if response_list else []
                if isinstance(last, list):
                    last = last[-1] if last else {}
                if isinstance(last, dict):
                    answer = last.get("content", "生成失败")
                else:
                    answer = str(last)
                if not answer or not str(answer).strip():
                    answer = "生成失败（模型没有返回内容）"

                st.subheader("🤖 AI综合回答")
                st.success(answer)
            else:
                st.warning("未检索到相关内容，无法生成回答。")
