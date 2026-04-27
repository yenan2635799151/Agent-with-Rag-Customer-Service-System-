from langchain_community.chat_message_histories import RedisChatMessageHistory

# 连接到Redis
chat_history = RedisChatMessageHistory(
    session_id="test_session_123",  # 替换为你要查看的session_id
    url="redis://localhost:6379",
    key_prefix="chat_history:"
)

# 打印所有消息
print("=== 对话历史 ===")
for i, msg in enumerate(chat_history.messages):
    print(f"\n[{i}] Type: {msg.type}, Role: {msg.type}")
    print(f"Content: {msg.content}")