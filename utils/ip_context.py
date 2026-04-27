from contextvars import ContextVar

"""
用于存储和获取当前请求的IP地址的上下文变量
"""

request_ip = ContextVar("request_ip",default=None)