import importlib
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_loacation,
                                     get_user_id,get_current_month,fill_context_for_report,fetch_external_data)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory, ChatMessageHistory
import json
import pickle
import os


def create_agent(*, model, system_prompt, tools, middleware=None):
    """
    兼容不同版本的 Agent 工厂函数：
    1) 优先使用 langchain.agents.create_agent
    2) 回退到 langgraph.prebuilt.create_react_agent
    """
    try:
        agents_module = importlib.import_module("langchain.agents")
        lc_create_agent = getattr(agents_module, "create_agent")
        return lc_create_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            middleware=middleware or []
        )
    except (ModuleNotFoundError, AttributeError):
        prebuilt_module = importlib.import_module("langgraph.prebuilt")
        lg_create_react_agent = getattr(prebuilt_module, "create_react_agent")
        return lg_create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt
        )


class FileChatMessageHistory:
    """
    文件存储的消息历史（作为 Redis 不可用时的备选）
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history_file = f"./chat_history_{session_id}.pkl"
        self.messages = []
        self.load_history()
    
    def load_history(self):
        """从文件加载历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'rb') as f:
                    self.messages = pickle.load(f)
            except:
                self.messages = []
    
    def save_history(self):
        """保存历史到文件"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(self.messages, f)
        except:
            pass
    
    def add_message(self, message):
        """添加消息"""
        self.messages.append(message)
        self.save_history()
    
    def clear(self):
        """清空历史"""
        self.messages = []
        if os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
            except:
                pass
    
    @property
    def messages(self):
        return self._messages
    
    @messages.setter
    def messages(self, value):
        self._messages = value


class ReactAgentWithMemory:
    def __init__(self, session_id: str):
        """
        初始化带持久化存储的 Agent
        
        Args:
            session_id: 会话 ID，用于存储和获取对话历史
        """
        self.session_id = session_id
        self.use_redis = False
        
        # 尝试使用 Redis 存储，如果失败则降级到文件存储
        try:
            self.chat_history = RedisChatMessageHistory(
                session_id=session_id,
                url="redis://localhost:6379",
                key_prefix="chat_history:"
            )
            # 测试连接
            self.chat_history.messages
            self.use_redis = True
            print(f"[Agent] 使用 Redis 存储会话历史 (session_id: {session_id})")
        except Exception as e:
            print(f"[Agent] Redis 连接失败 ({e})，降级到文件存储")
            self.chat_history = FileChatMessageHistory(session_id)
            print(f"[Agent] 使用文件存储会话历史 (session_id: {session_id})")
        
        # 初始化对话内存
        self.memory = ConversationBufferMemory(
            chat_memory=self.chat_history,
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化 Agent
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize,get_weather,get_user_loacation,get_user_id,
                   get_current_month,fill_context_for_report,fetch_external_data],
            middleware=[monitor_tool,log_before_model,report_prompt_switch]
        )

    def execute_stream(self, query: str):
        """
        执行流式回答
        
        Args:
            query: 用户查询
        
        Yields:
            回答的每一个字符
        """
        # 从内存中加载历史消息
        chat_history = self.memory.load_memory_variables({})
        
        # 构建输入字典
        input_dict = {
            "messages": chat_history["chat_history"] + [
                {"role": "user", "content": query}
            ]
        }
        
   
        # 执行 Agent 流式输出
        response_chunks = []  # 存储所有chunk
        full_response = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip() + "\n"
                response_chunks.append(content)
                full_response += content
                yield content
        
        # 只保存最后一个部分，避免重复
        if response_chunks:
            final_response = response_chunks[-1]
        else:
            final_response = full_response
        
        # 保存对话到内存
        self.memory.save_context(
            {"input": query},
            {"output": final_response}
        )

    def clear_history(self):
        """
        清空对话历史
        """
        self.chat_history.clear()

if __name__ == "__main__":
    # 测试带内存的 Agent
    agent = ReactAgentWithMemory("test_session_123")
    
    print("测试第一轮对话:")
    for chunk in agent.execute_stream("你好，我是小明"):
        print(chunk, end="", flush=True)
    
    print("\n\n测试第二轮对话:")
    for chunk in agent.execute_stream("我叫什么名字？"):
        print(chunk, end="", flush=True)
    
    # 测试不同会话
    agent2 = ReactAgentWithMemory("test_session_456")
    print("\n\n测试不同会话:")
    for chunk in agent2.execute_stream("你好，我是小红"):
        print(chunk, end="", flush=True)
    
    print("\n\n测试原会话:")
    for chunk in agent.execute_stream("我刚才说了什么？"):
        print(chunk, end="", flush=True)
