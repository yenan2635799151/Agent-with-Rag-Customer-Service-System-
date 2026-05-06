from rank_bm25 import BM25Okapi
import jieba
from utils.config_handler import rag_conf
from langchain_core.documents import Document
from typing import List
from functools import lru_cache
"""
轻量化的bm25 利用两阶段检索思想
用第一阶段的相似度检索的topk结果作为第二阶段bm25的输入
避免了docs每次都要读取的时间占用和内存占用
"""
class BM25_retriever:
    def __init__(self):
        self.stopwords = {
            "的","是","了","一个","我们","你们",
            "这个","那个","什么","如何","怎么",
            "吗","呢","啊","吧","就"
        }

    #对传入语句进行jieba分词并去除停用词
    def tokenize(self,text:str)->str:
        words = jieba.cut(text)
        tokens = [
            w for w in words 
            if w.strip() and w not in self.stopwords
        ]
        return tokens
    
    @lru_cache(maxsize=1000)#缓存分词好的内容，避免每次提问都重复计算
    def tokenize_cached(self,text:str):
        return tuple(self.tokenize(text))

    def rerank(self,query:str,docs:List[Document]):
        tokenized_query = self.tokenize(query)
        tokenized_corpus = [
            self.tokenize(doc.page_content)for doc in docs
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        ranked_docs = sorted(
            zip(docs,scores),#将分数与文档配对
            key = lambda x:x[1], #按照分数进行排序，x[1]是配对中的第二的数据
            reverse = True #由大到小排序，降序
        )

        return [doc for doc,_ in ranked_docs]#只做排序不做top_k筛选