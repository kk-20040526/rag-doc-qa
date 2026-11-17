# -*- coding: utf-8 -*-
"""
Elasticsearch 检索模块
支持海量文档的全文检索，基于 BM25 相关性排序
"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ESRetrieval:
    def __init__(self, host="http://localhost:9200", index_name="doc_index"):
        self.es = Elasticsearch([host])
        self.index_name = index_name
        try:
            self.es.indices.get(index=self.index_name)
        except Exception:
            self._create_index()

    def _create_index(self):
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "source": {"type": "keyword"}
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)

    def add_documents(self, chunks):
        """批量导入文档片段"""
        actions = []
        for i, chunk in enumerate(chunks):
            action = {
                "_index": self.index_name,
                "_id": i,
                "_source": chunk
            }
            actions.append(action)
        if actions:
            bulk(self.es, actions)
            self.es.indices.refresh(index=self.index_name)

    def search(self, query, top_k=5):
        """检索 Top-K 相关文档片段"""
        search_body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }
        res = self.es.search(index=self.index_name, body=search_body)
        return [
            {
                "content": hit["_source"]["content"],
                "source": hit["_source"]["source"],
                "score": hit["_score"]
            }
            for hit in res["hits"]["hits"]
        ]

    def delete_index(self):
        """删除索引"""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            return True
        return False
