from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.path_tool import get_abs_path
import  os
from utils.file_hander import txt_loader,pdf_loader,get_file_md5_hex,listdir_with_allowed_type
from utils.logger_handler import logger
from langchain_core.documents import Document
from rag.rag_tools.bm25 import BM25_retriever
from langchain_core.runnables import RunnableLambda
from rag.rag_tools.hybrid import HybridRetriever
from rag.rag_tools.CE_reranker import Reranker
from utils.config_handler import rag_conf
class VectorStoreService:
    def __init__(self):
        self.vector_store =Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"]
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chroma_conf["chunk_size"],
            chunk_overlap = chroma_conf["chunk_overlap"],
            length_function = len,
            separators = chroma_conf["separators"]
        )



    
    def load_documents(self):
        """
        从数据文件夹内读取文件，转为向量存入数据库，
        要计算文件的MD5做去重
        """

        def check_md5_hex(md5_for_cheak:str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]),"w").close() #创建空文件
                return False #md5 没处理过
            
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r",encoding="utf-8")as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_cheak:
                        return True #md5 处理过
                return False #MD5 没处理过
            
        def save_md5_hex(md5_for_save:str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding="utf-8")as f:# a 是追加模式，使用w会覆盖已经写了的内容
                f.write(md5_for_save+"\n")

        def get_file_documents(read_path:str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            else:
                return []
    
        allowed_files_path:list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}的内容已经存在知识库内，跳过")
                continue

            try:
                documents:list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}文件内没有有效文本内容，跳过")
                    continue
                split_document:list[Document] = self.splitter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}文件内的文本内容无法切分，跳过")
                    continue
               
                #将内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录已经处理好的文件的md5值，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]成功将{path}的内容加载到知识库内")

            except Exception as e:
                #exc_info=True 可以打印出错误的详细堆栈信息，方便调试，false只记录报错信息本身
                logger.error(f"[加载知识库]加载{path}文件时发生错误: {str(e)}",exc_info=True)
                continue

            
    def get_retriever(self):
        
        dense_retriever = self.vector_store.as_retriever(search_kwargs = {"k":chroma_conf["vector_recall_k"]})
        bm25_retriever = BM25_retriever()
        try: 
            reranker = Reranker()
            logger.info("成功加载reranker模型，hybrid retriever将启用reranker进行最终排序")
        except:
            reranker = None
            logger.info("加载reranker模型失败，hybrid retriever将不使用reranker进行最终排序")
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            vector_k=rag_conf["vector_k"],
            bm25_k=rag_conf["bm25_k"],
            final_k=rag_conf["final_k"]
        )
        return hybrid_retriever
    
    def load_chunks(self):
        data = self.vector_store.get()
        documents = data["documents"]
        metadatas = data["metadatas"]

        chunks = []
        for doc,meta in zip(documents,metadatas):
            if not doc or len(doc)<50:
                continue
            chunks.append({
                "content":doc,
                "metadata":meta
            })
        return chunks

if __name__ =="__main__":
    vs = VectorStoreService()
    vs.load_documents()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    print(len(res))
    for r in res :
        print(r.metadata)
        print(r.page_content)
        print("==="*20)