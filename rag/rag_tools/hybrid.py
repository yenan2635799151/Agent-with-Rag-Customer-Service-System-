from langchain_core.retrievers import BaseRetriever
from typing import Any, List
from langchain_core.documents import Document
from utils.config_handler import rag_conf
from rag.rag_tools.rrf import rrf_fusion
from rag.rag_tools.CE_reranker import Reranker
class HybridRetriever(BaseRetriever):
    """
    两阶段hybrid retriever
    Vector Recall → BM25 Rerank → (Optional) Reranker
    """
    dense_retriever: Any
    bm25_retriever: Any
    reranker:Any =None
    vector_k: int = rag_conf["vector_k"]
    bm25_k: int = rag_conf["bm25_k"]
    final_k: int = rag_conf["final_k"]
    

    class Config:
        arbitrary_types_allowed = True  # 👈 关键！

    def _get_relevant_documents(self,query:str)->List[Document]:
        #vector recall 筛选50
        candidates = self.dense_retriever.invoke(query)
        if not candidates:
            return []
        #控制候选数据数量
        candidates = candidates[:self.vector_k]
        bm25_ranked = self.bm25_retriever.rerank(query,docs = candidates)

        bm25_topk = bm25_ranked[:self.bm25_k]

        if self.reranker:
            ranked = self.reranker.rerank(query,bm25_topk)
        else:
            ranked = bm25_topk
        #最终返回topk结果
        return ranked[:self.final_k]