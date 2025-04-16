import os
import time
import redis
from flask import Flask, request, Response, stream_with_context
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# 配置参数
REDIS_URL = "redis://localhost:6379"  # 仅用于聊天历史
DOCS_PATH = "./doc"
FAISS_INDEX_PATH = "./faiss_index"  # 向量索引保存路径
EMBED_MODEL = "nomic-embed-text"  # 专用嵌入模型
LLM_MODEL = "deepseek-r1:8b"

# 初始化Redis（仅用于聊天历史）
redis_client = redis.Redis.from_url(REDIS_URL)

# 初始化模型
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url="http://localhost:11434"
)
llm = Ollama(model=LLM_MODEL, temperature=0, base_url="http://localhost:11434")


# 初始化向量库
def initialize_vectorstore():
    # 如果已有保存的索引则直接加载
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # 否则处理文档并创建索引
    documents = []
    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.endswith(".doc"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                continue

            documents.extend(loader.load())
            print(f"成功加载: {file}")
        except Exception as e:
            print(f"加载文件 {file} 失败: {str(e)}")
            continue

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    # 构建FAISS索引
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(FAISS_INDEX_PATH)
    return vector_db


# 自定义检索QA链
template = """使用以下上下文和聊天历史来回答最后的问题。如果你不知道答案，就说你不知道，不要编造答案。
上下文：{context}
聊天历史：{chat_history}
问题：{question}
答案："""
prompt = PromptTemplate.from_template(template)


def get_qa_chain():
    vector_db = initialize_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        }
    )


# 聊天历史处理（保持Redis存储）
def get_chat_history(session_id="default"):
    history_key = f"chat_history:{session_id}"
    return redis_client.lrange(history_key, 0, -1)


def save_to_history(session_id="default", question="", answer=""):
    history_key = f"chat_history:{session_id}"
    redis_client.lpush(history_key, f"User: {question}")
    redis_client.lpush(history_key, f"AI: {answer}")
    redis_client.ltrim(history_key, 0, 10)


# 流式生成响应
def generate_stream(query, session_id="default"):
    qa_chain = get_qa_chain()
    chat_history = get_chat_history(session_id)

    response = qa_chain.invoke({
        "query": query,
        "chat_history": "\n".join([h.decode() for h in chat_history])
    })

    save_to_history(session_id, query, response["result"])

    for word in response["result"].split():
        yield word + " "
        time.sleep(0.05)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    session_id = data.get('session_id', 'default')
    return Response(
        stream_with_context(generate_stream(question, session_id)),
        mimetype='text/event-stream'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)