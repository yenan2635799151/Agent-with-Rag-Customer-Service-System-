from abc import ABC,abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils.config_handler import rag_conf
import os
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)-> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self)-> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model = rag_conf["chat_model_name"],
            base_url=rag_conf["chat_model_url"],
            api_key=rag_conf["chat_model_api_key"],
        )
    
class EmbeddingsModelFactory(BaseModelFactory):
    def generator(self)-> Optional[Embeddings | BaseChatModel]:
        return HuggingFaceBgeEmbeddings(
            model_name=rag_conf["embedding_model_name"],
            model_kwargs=rag_conf["embedding_model_kwargs"]
        )
    
chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsModelFactory().generator()