from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from utils.path_tool import get_abs_path
from utils.file_hander import txt_loader, pdf_loader, word_loader, markdown_loader, csv_loader, get_file_md5_hex, listdir_with_allowed_type
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embed_model
from rag.rag_tools.bm25 import BM25_retriever
from rag.rag_tools.hybrid import HybridRetriever
from rag.rag_tools.CE_reranker import Reranker
from utils.config_handler import chroma_conf, rag_conf
import os
from typing import Any


class MilvusVectorStoreService:
    def __init__(self, collection_name="rag_collection"):
        connections.connect(host="localhost", port="19530")

        # 统一 embedding 结构后再取维度，避免返回值是嵌套列表导致 dim 计算错误。
        self.dim = len(self._normalize_embedding(embed_model.embed_query("测试")))

        self.collection_name = collection_name

        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
        ]
        self.schema = CollectionSchema(self.fields)

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            logger.info(f"[Milvus] 已存在 collection {collection_name}")

            existing_dim = self._get_collection_dim(self.collection)
            if existing_dim is not None and existing_dim != self.dim:
                logger.warning(
                    f"[Milvus] 维度不一致，重建 collection: existing_dim={existing_dim}, current_dim={self.dim}"
                )
                self.collection.release()
                utility.drop_collection(collection_name)
                self.collection = Collection(collection_name, schema=self.schema)
                logger.info(f"[Milvus] 已重建 collection {collection_name}")
        else:
            self.collection = Collection(collection_name, schema=self.schema)
            logger.info(f"[Milvus] 创建 collection {collection_name}")

        # ✅ 索引
        if len(self.collection.indexes) == 0:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)
            logger.info("[Milvus] 索引创建成功")

        # ✅ 必须始终 load
        self.collection.load()
        logger.info("[Milvus] Collection 加载完成")



        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"]
        )

    @staticmethod
    def _normalize_embedding(emb: Any) -> list[float]:
        if hasattr(emb, "tolist"):
            emb = emb.tolist()

        # 兼容 [[...]] 这类单条向量被多包一层的情况。
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            if len(emb) == 1:
                emb = emb[0]
            else:
                raise ValueError(f"embedding 结构异常，期望单向量，实际批量长度: {len(emb)}")

        if not isinstance(emb, list):
            raise TypeError(f"embedding 类型异常: {type(emb)}")

        return emb

    @staticmethod
    def _get_collection_dim(collection: Collection) -> int | None:
        for field in collection.schema.fields:
            if field.name == "embedding":
                return int(field.params.get("dim"))
        return None

    # -------------------------
    # 加载文档
    # -------------------------
    def load_documents(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w").close()  # 创建空文件
                return False  # md5 没处理过
            
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5 处理过
                return False  # MD5 没处理过
        
        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:  # a 是追加模式，使用w会覆盖已经写了的内容
                f.write(md5_for_save + "\n")

        allowed_files = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[Milvus] {path}的内容已经存在知识库内，跳过")
                continue

            try:
                if path.endswith("txt"):
                    docs = txt_loader(path)
                elif path.endswith(".pdf"):
                    docs = pdf_loader(path)
                elif path.endswith(".docx") or path.endswith(".doc"):
                    docs = word_loader(path)
                elif path.endswith(".md") or path.endswith(".markdown"):
                    docs = markdown_loader(path)
                elif path.endswith(".csv"):
                    docs = csv_loader(path)
                else:
                    logger.info(f"[Milvus] 不支持的文件类型: {path}")
                    continue

                split_docs = self.splitter.split_documents(docs)

                if not split_docs:
                    logger.info(f"[Milvus] 文件无可插入切片，跳过: {path}")
                    continue

                texts = [doc.page_content for doc in split_docs]

                # 批量 embedding，减少重复调用并保证输出结构稳定。
                raw_embeddings = embed_model.embed_documents(texts)
                embeddings = []
                texts = []

                for i, emb in enumerate(raw_embeddings):
                    emb = self._normalize_embedding(emb)

                    if len(emb) != self.dim:
                        raise ValueError(f"维度错误: {len(emb)} != {self.dim}")

                    embeddings.append(emb)
                    texts.append(split_docs[i].page_content)

                # ✅ 插入
                self.collection.insert([embeddings, texts])
                logger.info(f"[Milvus] 插入成功: {path}")

                # 记录已经处理好的文件的md5值，避免下次重复加载
                save_md5_hex(md5_hex)

            except Exception as e:
                logger.error(f"[Milvus] 加载失败: {e}", exc_info=True)

        # ✅ flush（确保可查）
        self.collection.flush()

    # -------------------------
    # Retriever
    # -------------------------
    def get_retriever(self):
        dense = MilvusRetriever(self.collection, embed_model, k=rag_conf["vector_k"])
        bm25 = BM25_retriever()

        try:
            reranker = Reranker()
        except:
            reranker = None

        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            reranker=reranker,
            vector_k=rag_conf["vector_k"],
            bm25_k=rag_conf["bm25_k"],
            final_k=rag_conf["final_k"]
        )


class MilvusRetriever:
    def __init__(self, collection, embed_model, k=5):
        self.collection = collection
        self.embed_model = embed_model
        self.k = k

    def get_relevant_documents(self, query):
        emb = MilvusVectorStoreService._normalize_embedding(self.embed_model.embed_query(query))

        results = self.collection.search(
            data=[emb],
            anns_field="embedding",
            param={"nprobe": 10},
            limit=self.k,
            output_fields=["text"]
        )

        docs = []
        for hit in results[0]:
            docs.append(
                Document(
                    page_content=hit.entity.get("text"),
                    metadata={"score": hit.distance}
                )
            )
        return docs

    def invoke(self, query, **kwargs):
        return self.get_relevant_documents(query)


# -------------------------
# 测试
# -------------------------
if __name__ == "__main__":
    vs = MilvusVectorStoreService()

    vs.load_documents()

    retriever = vs.get_retriever()

    query = "迷路"
    res = retriever.invoke(query)

    print(f"召回数量: {len(res)}")

    for r in res:
        print(r.metadata)
        print(r.page_content[:100])
        print("===" * 20)