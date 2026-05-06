
from typing import List
from langchain_core.documents import Document
from collections import defaultdict
def rrf_fusion(results_list,k=60):
    """
    results_list: List[List[Document]]
    k:RRF参数，越小越重视排名靠前的结果，越大越平均
    """
    scores = defaultdict(float)
    doc_map ={}
    
    for result in results_list:
        for rank,doc in enumerate(result):
            key = doc.page_content

            #RFF计算公式 score = 1/(k+rank)
            scores[key] += 1/(k+rank)
            doc_map[key] = doc

    #根据RRF分数对文档进行排序
    sorted_docs = sorted(
        scores.items(),
        key = lambda x:x[1],
        reverse=True  
    )
    return [doc_map[key] for key,_ in sorted_docs]