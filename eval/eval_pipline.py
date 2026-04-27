import json
from tqdm import tqdm
from datasets import Dataset
from utils.config_handler import chroma_conf, rag_conf
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from model.factory import chat_model,embed_model
from utils.config_handler import chroma_conf, rag_conf
from rag.vector_store import VectorStoreService
import os
from utils.path_tool import get_abs_path
import re
from typing import Any

import re
import json


    

llm = chat_model

# 定义一个包装类，用于清理 LLM 输出中的 Markdown 代码块标记
class CleanMarkdownLLM(ChatOpenAI):
    def generate(self, prompts, *args, **kwargs):
        # 调用父类的 generate 方法
        result = super().generate(prompts, *args, **kwargs)
        
        # 清理每个生成结果中的 Markdown 代码块标记
        for i, generation in enumerate(result.generations):
            for j, gen in enumerate(generation):
                if gen.text.startswith('```json'):
                    # 清理 Markdown 代码块标记
                    cleaned = re.sub(r'^```json\s*|\s*```$', '', gen.text, flags=re.DOTALL)
                    gen.text = cleaned
        
        return result

eval_llm = CleanMarkdownLLM(
    model=rag_conf["chat_model_name"],
    base_url=rag_conf["chat_model_url"],
    api_key=rag_conf["chat_model_api_key"],
    temperature=0,
    n=1,
    model_kwargs={
        "response_format": {"type": "json_object"}   # 👈 关键！！
    }
)


metrics = [faithfulness, answer_relevancy]
for m in metrics:
    m.llm = eval_llm   

    
#生成query（llm）
def build_query_prompt(chunk):
    return f"""
你是一个用户问题生成器。

请基于下面的知识内容，生成1个用户可能会问的问题：

要求：
- 问题自然
- 不要照抄原文
- 简洁具体
- 中文

知识：
{chunk}

只返回问题
"""

def clean_markdown_json(text):
    """清理 Markdown 代码块标记，只保留纯 JSON 内容"""
    if text.startswith('```json'):
        # 移除 Markdown 代码块标记
        cleaned = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.DOTALL)
        return cleaned
    return text

def generate_query(chunks):
    prompt = build_query_prompt(chunks)
    answer = llm.invoke(prompt)
    print(type(answer))
    print(answer)
    print("==="*20)
    res = answer.content.strip()
    # 清理可能的 Markdown 代码块标记
    res = clean_markdown_json(res)
    return str(res).strip()



#RAG Pipline

def rag_pipeline(query,retriever,llm):
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"""
请基于以下内容回答问题
{context}

问题：{query}
"""
    
    answer = llm.invoke(prompt)
    answer = answer.content.strip()
    # 清理可能的 Markdown 代码块标记
    answer = clean_markdown_json(answer)
    answer = str(answer).strip()
    return{
        "answer":answer,
        "docs":docs
    }

#构建数据评估
def build_dataset(chunks,retriever,llm,max_samples=30):
    data = []
    
    for c in tqdm(chunks[:max_samples],desc="生成评估数据"):
        try:
            #生成问题
            question = generate_query(c["content"])

            #跑rag
            result = rag_pipeline(question,retriever,llm)

            data.append({
                "question":question,
                "answer":result["answer"],
                "contexts":[d.page_content for d in result["docs"]]
            })

        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue
    return data


#RAGAS 评估

def run_ragas(data):
    dataset = Dataset.from_list(data)

    result = evaluate(
        dataset,
        metrics = metrics,
        llm = eval_llm,
        embeddings=embed_model,
    )

    return result


def main():
    print("初始化向量库")
    vs_serviec = VectorStoreService()
    retriever = vs_serviec.get_retriever()
    print("加载chunks")
    chunks = vs_serviec.load_chunks()
    print(f"chunks数量:{len(chunks)}")

    data_file = "ragas_dataset.json"
    abs_data_file = get_abs_path(data_file)
    if os.path.exists(abs_data_file):
        print(f"找到现有评估数据文件{abs_data_file}，直接加载")
        with open(abs_data_file,"r",encoding="utf-8") as f:
            data = json.load(f)
    else:
        print("构建评估数据")
        data = build_dataset(chunks,retriever,llm,max_samples=30)

        with open(abs_data_file,"w",encoding="utf-8")as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    print("运行RAGAS评估")
    result = run_ragas(data)

    print("评估结果:")
    print(f"样本数量：{len(data)}")
    print(result)

if __name__ =="__main__":
    #res = eval_llm.invoke("输出一个JSON：{'a':1}")
   # print(res.content)
    main()
