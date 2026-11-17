# -*- coding: utf-8 -*-
"""BM25 知识库检索工具（不依赖 qwen-agent 的 Retrieval 接口）。

为什么需要这个文件：
    qwen-agent 从 0.0.9 到 0.0.34，其内置 Retrieval 工具的参数都是
        {"query": str, "files": [文件路径...]}
    也就是每次调用都要传一批真实文件路径，由 DocParser 现场解析。
    而 final_app.py 里的写法是
        Retrieval({"docs": docs, "method": "bm25"})
    —— 把段落直接塞进配置。这个接口在当前所有可安装版本里都不存在，
    结果是 call() 走 files=[] 分支返回 []，检索永远为空，模型只能凭自己答。

用法（replace 掉 final_app.py 里的 kb 构造即可，其余代码不用动）：
    from kb_tool import BM25KBTool
    kb = BM25KBTool({"docs": docs})
    bot = Assistant(llm=llm_cfg, system_message=..., function_list=[kb])

设计说明：
    - 用 rank_bm25 直接建索引，jieba 分词，不依赖向量库、不解析文件；
    - 返回带来源和得分的文本块，模型能据此引用，界面也能展示依据；
    - 与框架解耦，qwen-agent 升级不会再打断检索。
"""
from typing import Dict, Optional, Union

import jieba
from qwen_agent.tools.base import BaseTool, register_tool

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover
    raise ImportError('需要 rank_bm25：pip install "qwen-agent[rag]"') from exc


# 停用词表：中文虚词几乎没有区分能力，却几乎每段都有。
# 不去掉它们会出两种坏情况：① 术语被"的/是/什么"的分数淹没；
# ② 短片段因为长度归一化反而赢过长片段。实测就是"问 PaddleOCR 算法"
# 却捞出《文档切分策略》——那篇里一个"识别/SVTR"都没有。
STOPWORDS = {
    '的', '了', '是', '在', '有', '和', '与', '或', '也', '就', '都', '而', '及', '其',
    '这', '那', '这个', '那个', '一个', '我们', '你', '我', '他', '它', '你们', '什么',
    '怎么', '如何', '为什么', '用', '吗', '呢', '吧', '啊', '把', '被', '从', '到',
    '对', '为', '以', '于', '上', '下', '之', '等', '中', '时', '可以', '需要', '进行',
    '通过', '因为', '所以', '如果', '但是', '不', '没', '没有', '会', '要', '想',
    '请问', '一下', '具体', '分别', '区别', '哪些', '哪个', '多少', '介绍', '说明',
}


def _tokenize(text: str) -> list:
    """jieba 分词 + 停用词过滤。

    只滤停用词，不滤单个汉字：'搜'（搜不到结果）、'帧'、'层' 这类单字是实义词，
    滤掉会把"为什么 ES 搜不到结果"这种问题正好推错文档（实测过）。
    """
    out = []
    for tok in jieba.cut(text):
        tok = tok.strip().lower()
        if not tok or tok in STOPWORDS:
            continue
        if not any(ch.isalnum() for ch in tok):
            continue  # 纯标点（如全角"？"、"。"）不参与打分
        out.append(tok)
    return out


@register_tool('bm25_kb')
class BM25KBTool(BaseTool):
    """在本地文档片段上做 BM25 检索。"""

    description = ('从本地知识库中检索与问题最相关的文档片段。'
                   '任何时候只要用户提出问题，都应该先调用本工具获取依据，再回答。')
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'description': '用于检索的关键词或问题，中英文均可，多个关键词用逗号分隔。',
                'type': 'string',
            },
        },
        'required': ['query'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.docs = (cfg or {}).get('docs') or []
        self.top_k = (cfg or {}).get('top_k', 5)
        self.expand = (cfg or {}).get('expand_neighbors', 1)  # 命中片段附带前后邻居
        self._tokens = [_tokenize(d['content']) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None
        # 同源文档内的相邻关系（同一 source 的片段按原顺序编号）
        self._neighbors = []
        for i, d in enumerate(self.docs):
            prev_i = i - 1 if i > 0 and self.docs[i - 1].get('source') == d.get('source') else None
            next_i = (i + 1 if i + 1 < len(self.docs)
                      and self.docs[i + 1].get('source') == d.get('source') else None)
            self._neighbors.append((prev_i, next_i))

    def _format(self, hits) -> str:
        if not hits:
            return '未检索到相关内容。'
        blocks = []
        for rank, (idx, score) in enumerate(hits, 1):
            d = self.docs[idx]
            body = d['content']
            # 邻接扩展：切分容易把"问的术语"和"答的术语"切到两段里，
            # 命中一段时把同源的上一段/下一段一起带上，缓解切断。
            if self.expand:
                prev_i, next_i = self._neighbors[idx]
                if prev_i is not None:
                    body += '\n（上一段）' + self.docs[prev_i]['content']
                if next_i is not None:
                    body += '\n（下一段）' + self.docs[next_i]['content']
            blocks.append('【片段%d｜来源：%s｜BM25得分：%.2f】\n%s'
                          % (rank, d.get('source', 'unknown'), score, body))
        return '\n\n'.join(blocks)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params.get('query', '')
        if not self._bm25 or not query:
            return '未检索到相关内容。'

        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        order = scores.argsort()[::-1]
        hits = [(int(i), float(scores[i])) for i in order if scores[i] > 0][:self.top_k]
        return self._format(hits)
