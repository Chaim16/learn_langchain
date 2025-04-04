import os

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.conf_util import openai_config

openai_config = openai_config()
os.environ["OPENAI_API_KEY"] = openai_config.get("api_key")

session_store = {}


def get_chat_history(session_id: str):
    return session_store.setdefault(session_id, ChatMessageHistory())


# 构建三级对话模板
conversation_blueprint = ChatPromptTemplate([
    ("system", "你是一个专业的人工智能助手"),  # 角色定义
    MessagesPlaceholder(variable_name="history"),  # 历史记忆
    ("human", "{input}"),  # 用户输入
])

# 初始化大预言模型
llm_engine = ChatOpenAI(model="gpt-40-mini", max_tokens=1000, temperature=0.9)

# 构建处理链
processing_pipeline = conversation_blueprint | llm_engine

# 添加历史管理模块
smart_agent = RunnableWithMessageHistory(
    processing_pipeline,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

user_session = "user_001"
print("aaaaa")
first_response = smart_agent.invoke(
    {"input": "你好，可以写一个冒泡排序算法吗？"},
    config={"configurable": {"session_id": user_session}},
)
print(f"AI回复：{first_response}")

# 延续对话
followup_response = smart_agent.invoke(
    {"input": "我刚才问的问题是什么？"},
    config={"configurable": {"session_id": user_session}},
)
print(f"AI回复：{followup_response}")


# 查看对话历史
print("\n完整对话记录：")
for message in session_store[user_session].messages:
    print(f"[{message.type.upper()}]  {message.content}")

