import os
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama  # 使用 OllamaChat 来调用本地模型
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# 创建本地模型实例
llm = ChatOllama(model_name="MHKetbi/Mistral-Small3.1-24B-Instruct-2503:q4_K_L", temperature=0.7)  # 这里是使用 Ollama 本地模型
# 创建一个简单的对话记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 加载文档
loader = TextLoader("docs/a.txt", encoding="utf-8")
documents = loader.load()
# 生成文档的嵌入向量
embeddings = OllamaEmbeddings()  # 可以继续使用 OpenAI 的嵌入，或者使用其他的嵌入模型
vectorstore = FAISS.from_documents(documents, embeddings)
# 使用 FAISS 向量数据库作为检索源
retriever = vectorstore.as_retriever()

# 创建一个检索-问答链（Retrieval-QA Chain）
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def chat_with_memory(user_input):
    """处理用户输入，记住对话历史，并返回AI回复"""

    # 从记忆中获取上下文并更新回答
    context = memory.load_memory_variables({})["chat_history"]
    response = qa_chain.run(user_input + "\nContext: " + str(context))

    # 返回答案并更新记忆
    memory.save_context({"input": user_input},{"output": response})
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
