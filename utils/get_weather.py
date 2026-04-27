import requests
from utils.logger_handler import logger
from utils.config_handler import agent_conf
def _get_city_code(city:str)->str:
    """
    内部函数：根据城市名称获取城市代码
    """
    city_map = {
        "北京": "110000",
        "上海": "310000",
        "广州": "440100",
        "深圳": "440300",
        "杭州": "330100",
        "成都": "510100",
        "重庆": "500000",
        "武汉": "420100",
        "西安": "610100",
        "南京": "320100",
    }
    return city_map.get(city,"110000")  # 默认北京

def get_city_weather(city:str)->str:
    """
    调用高德api获取天气信息的工具，以字符串的形式返回
    """
    try:
        api_key = agent_conf["get_weather_api_key"]
        if not api_key:
            logger.error(f"[get_weather]未配置高德地图API key，无法获取天气信息")
            return None
        city_code = _get_city_code(city)

        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params={
            "key":api_key,
            "city":city_code,
            "extensions":"base"
        }

        res = requests.get(url,params=params,timeout=2).json()

        if res.get("status")!="1":
            logger.error(f"[get_weather]高德地图API请求失败，返回：{res}")
            return None
        
        info = res["lives"][0]

        weather = info.get("weather")
        temperature = info.get("temperature")
        humidity = info.get("humidity")
        winddirection = info.get("winddirection")
        windpower = info.get("windpower")
        reporttime = info.get("reporttime")

        return (
            f"{city}当前天气：{weather}，气温{temperature}°C，"
            f"湿度{humidity}%，{winddirection}风{windpower}级，"
            f"更新时间：{reporttime}"
        )
    
    except Exception as e:
        logger.error(f"[get_weather]获取天气信息时发生错误：{str(e)}")
        return None
