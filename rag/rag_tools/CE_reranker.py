from sentence_transformers import CrossEncoder
from utils.config_handler import rag_conf

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(rag_conf["reranker_model_name"])

    def rerank(self, query, docs):
        pairs = [(query,doc.page_content)for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
        return [doc for doc,_ in ranked]