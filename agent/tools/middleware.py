from langchain.agents.middleware import wrap_tool_call,before_model,dynamic_prompt,ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from utils.prompt_loader import load_report_prompt,load_system_prompt
@wrap_tool_call
def monitor_tool(
    #请求的数据的封装
    request: ToolCallRequest,
    #执行的函数本身
    handler:Callable[[ToolCallRequest],ToolMessage|Command],
)-> ToolMessage|Command:            #工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        res = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}执行成功")

        if request.tool_call['name']=="fill_context_for_report":
            request.runtime.context["report"]=True
        return res
    except Exception as e:
        logger.error(f"[tool monitor]工具{request.tool_call['name']}执行失败,原因：{str(e)}")
        raise e
@before_model
def log_before_model(
        state:AgentState,#整个agent智能体中的状态记录
        runtime:Runtime,#记录了整个执行过程中的上下文信息
): #模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__}|当前消息：{state['messages'][-1].content.strip()}")#用[-1]取最后一条最新的消息
    
    return None
    
@dynamic_prompt #每一次在生成提示词前调用此函数
def report_prompt_switch(request:ModelRequest):#动态切换提示词
    is_rport = request.runtime.context.get("report",False)
    if is_rport: #需要生成报告，返回报告生成提示词内容
        return load_report_prompt()
    
    return load_system_prompt()

