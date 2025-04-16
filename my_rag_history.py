import os
import redis
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify, Response
import json

app = Flask(__name__)

# 连接 Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 初始化嵌入模型（建议使用中文模型）
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 加载文档（带详细日志）
def load_knowledge_base(directory):
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            ext = file.lower()
            try:
                if ext.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif ext.endswith(('.docx', '.doc')):
                    loader = Docx2txtLoader(path)
                elif ext.endswith('.txt'):
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    print(f"[警告] 跳过不支持的格式：{file}")
                    continue

                loaded = loader.load()
                valid_docs = [d for d in loaded if d.page_content.strip()]
                if valid_docs:
                    print(f"[信息] 加载 {len(valid_docs)} 个文档：{file}")
                    docs.extend(valid_docs)
                else:
                    print(f"[警告] 空文档：{file}")

            except Exception as e:
                print(f"[错误] 加载 {file} 失败：{e}")
    if not docs:
        raise ValueError("知识库为空，请检查目录和文件")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return FAISS.from_documents(splitter.split_documents(docs), embeddings)

# 替换为你的文档目录（必须包含相关文件）
KB_DIR = "./doc"
db = load_knowledge_base(KB_DIR)

# 初始化 LLM
llm = OllamaLLM(
    model="deepseek-r1:8b",
    callbacks=[StreamingStdOutCallbackHandler()],
    base_url="http://localhost:11434"
)

# 自定义检索链（处理无文档情况）
class CustomRetrievalQA(RetrievalQA):
    def _call(self, inputs, run_manager=None):
        query = inputs['query']
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return {"result": "对不起，当前知识库中没有相关信息，请提供更多背景或查阅具体文档。"}
        return super()._call(inputs, run_manager)

qa = CustomRetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 聊天历史管理
def save_history(user_id, q, a):
    redis_client.rpush(f"chat:{user_id}", f"User: {q}")
    redis_client.rpush(f"chat:{user_id}", f"Bot: {a}")

def get_history(user_id):
    return [m.decode() for m in redis_client.lrange(f"chat:{user_id}", 0, -1)]

# API 端点
@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    user_id = data.get('user_id')
    question = data.get('question')

    history = get_history(user_id)
    context = "\n".join(history)
    prompt = f"历史对话：{context}\n当前问题：{question}\n请结合知识库回答，若需具体文档信息请说明。"

    def generate():
        try:
            result = qa.invoke({"query": prompt})
            answer = result["result"]
            for char in answer:
                yield json.dumps({"chunk": char}) + '\n'
            save_history(user_id, question, answer)
        except Exception as e:
            yield json.dumps({"error": str(e)}) + '\n'

    return Response(generate(), mimetype='application/json')

# 新增：清空 Redis 历史记录的接口
@app.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    user_id = data.get('user_id')
    if user_id:
        try:
            redis_client.delete(f"chat:{user_id}")
            return jsonify({"message": f"用户 {user_id} 的聊天历史记录已清空"})
        except Exception as e:
            return jsonify({"error": f"清空历史记录时出错：{str(e)}"}), 500
    else:
        return jsonify({"error": "请求中缺少 user_id 参数"}), 400


if __name__ == '__main__':
    app.run(debug=False, port=5000)
