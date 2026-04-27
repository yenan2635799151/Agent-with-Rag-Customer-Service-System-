from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import  random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import os
from utils.logger_handler import logger
from utils.get_loc import _get_loction_from_ip
from utils.ip_context import request_ip
from utils.get_weather import get_city_weather
rag = RagSummarizeService()
user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]
month_arr=["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12"]
external_data = {}

@tool 
def rag_summarize(query:str)->str:
    """
    根据用户提问从向量存储中检索参考资料
    """
    return rag.rag_summarize(query)

@tool
def get_weather(city:str)->str:
    """
    获取指定城市的天气信息的工具，以字符串的形式返回
    """
    res = get_city_weather(city)
    if res == None:
        logger.warning(f"[get_weather]未能获取到城市{city}的天气信息，返回默认天气信息")
        return f"城市{city}的天气为晴天，气温26摄氏度，空气湿度50%,南风1级，AQI21，最近六小时降雨概率极低。"
    else:
        return res
@tool
def get_user_loacation():
    """
    获取用户所在城市的名称，以纯字符串形式返回

    """
    ip = request_ip.get()
    if not ip:
        logger.warning(f"[get_user_location]无法获取用户的ip地址，无法获取用户的位置信息")
        return random.choice(["北京","上海","广州","深圳","杭州"]) 

    if ip in ["127.0.0.1", "localhost", "::1"]:
        logger.warning(f"[get_user_location]获取到用户的ip地址为本地地址{ip}使用预设地址")
        return random.choice(["北京","上海","广州","深圳","杭州"])
    
    res = _get_loction_from_ip(ip)
    
    if(res=="未知城市"):
        logger.warning(f"[get_user_location]获取到用户的ip地址为{ip}但无法获取城市信息，使用预设地址")
        return random.choice(["北京","上海","广州","深圳","杭州"])
    else:   
        return res

@tool
def get_user_id()->str:
    """
    获取用户ID，以纯字符串形式返回
    """
    return random.choice(user_ids)

@tool 
def get_current_month()->str:
    """
    获取当前的月份，以纯字符的形式输出
    """
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id":{
            "month":{"特征":xxx,"效率":xxx ,...}
            "month":{"特征":xxx,"效率":xxx ,...}
            "month":{"特征":xxx,"效率":xxx ,...}
            "month":{"特征":xxx,"效率":xxx ,...}
        }
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        with open(external_data_path,"r",encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr:list[str] = line.strip().split(",")

                user_id:str = arr[0].replace('"',"")
                feature:str = arr[1].replace('"',"")
                efficiency:str = arr[2].replace('"',"")
                consumables:str = arr[3].replace('"',"") 
                comparison:str = arr[4].replace('"',"")
                time:str = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time]={
                    "特征":feature,
                    "效率":efficiency,
                    "耗材":consumables,
                    "对比":comparison
                }
@tool
def fetch_external_data(user_id:str,month:str)->str:
    """
    从外部系统获取指定用户和月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串
    """
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户{user_id}在{month}的使用记录数据")
        return ""
    
@tool
def fill_context_for_report():
    """
    无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词提供上下文信息
    """
    return "fill_context_for_report已调用"