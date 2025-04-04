import os

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from utils import conf_util

openai_config = conf_util.openai_config()
os.environ["OPENAI_API_KEY"] = openai_config.get("api_key")

# 创建 OpenAI 模型实例
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# 创建一个简单的对话记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 加载文档
loader = TextLoader("docs/a.txt", encoding="utf-8")
documents = loader.load()
# 生成文档的嵌入向量
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
# 使用 FAISS 向量数据库作为检索源
retriever = vectorstore.as_retriever()

# 创建一个检索-问答链（Retrieval-QA Chain）
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def chat_with_memory(user_input):
    """处理用户输入，记住对话历史，并返回AI回复"""

    # 用户输入添加到记忆中
    memory.add_message(HumanMessage(content=user_input))

    # 从记忆中获取上下文并更新回答
    context = memory.load_memory_variables({})["chat_history"]
    response = qa_chain.run(user_input, "\nContext: " + str(context))

    # 返回答案并更新记忆
    memory.add_message(HumanMessage(content=response))
    return response


if __name__ == '__main__':
    print("欢迎使用智能问答系统！")
    while True:
        user_input = input("你：")
        if user_input.lower() in ["退出", "exit", "bye"]:
            print("AI: 再见！")
            break
        response = chat_with_memory(user_input)
        print("AI:", response)
    print(os.environ["OPENAI_API_KEY"])
