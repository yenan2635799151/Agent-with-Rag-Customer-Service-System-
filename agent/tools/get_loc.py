from langchain.tools import tool
import requests
from functools import lru_cache
from utils.logger_handler import logger

@lru_cache(maxsize=128)
def _get_loction_from_ip(ip:str)->str:
    """
    内部函数：根据ip查询城市（带缓存）
    """
    try:
        #免费IP定位API
        url = f"http://ip-api.com/json/{ip}?lang=zh-CN"
        res = requests.get(url,timeout=2).json()

        if res.get("status")=="success":
            city = res.get("city")
            region = res.get("regionName")
            country= res.get("country")

            #优先返回城市
            if city :
                return city
            elif region:
                return region
            elif country:
                return country
            
        return "未知城市"
    except Exception:
        logger.error(f"[get_location]获取城市信息失败{ip}无法获取城市信息")
        return "未知城市"
