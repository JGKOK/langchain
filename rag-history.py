from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

# ================== 配置参数 ==================
MODEL_NAME = "deepseek-r1:8b"
HISTORY_FILE = "chat_history.json"
HISTORY_WINDOW = 3

# ================== 新版Prompt模板 ==================
DEEPSEEK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """严谨技术文档分析要求：
1. 分点回答（★标记）
2. 标注页码（示例：★内容...(p.12)）
3. 使用中文作答"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),  # 统一使用input作为输入键
])

# ================== 修复后的RAG系统 ==================
class DeepSeekRAGSystem:
    def __init__(self):
        self.vector_db = self.init_vector_db()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=HISTORY_WINDOW,
            return_messages=True
        )
        self.chain = self.build_chain()

    def init_vector_db(self):
        """初始化向量数据库"""
        loader = DirectoryLoader(
            "./doc", 
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        texts = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            separators=["\n\n", "。", "\n"]
        ).split_documents(loader.load())
        
        return FAISS.from_documents(
            texts,
            OllamaEmbeddings(model="nomic-embed-text")
        )

    def build_chain(self):
        """构建符合新版API的问答链"""
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        # 历史感知检索器
        history_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "根据对话历史重新组织问题")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=Ollama(model=MODEL_NAME),
            retriever=retriever,
            prompt=history_prompt
        )

        # 主问答链
        return (
            RunnablePassthrough.assign(
                context=history_aware_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
            )
            | DEEPSEEK_PROMPT
            | Ollama(model=MODEL_NAME, temperature=0.3)
        )
        
    def save_history(self):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.memory.load_memory_variables({}), f)

    def load_history(self):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []    

    def chat_loop(self):
        """修复后的对话循环"""
        print("DeepSeek RAG系统已启动（输入'退出'结束）")
        while True:
            query = input("\n用户提问：")
            if query.lower() in ['退出', 'exit']:
                print("再见！")
                break
            
            response = self.chain.invoke({
                "input": query,  # 统一使用input作为输入键
                "chat_history": self.memory.buffer
            })
            
            self.memory.save_context({"input": query}, {"output": response})
            
            print("\n" + "="*60)
            print(f"📌 问题：{query}")
            print("🔍 回答：\n" + response)
            print("="*60)

# ================== 主执行 ==================
if __name__ == "__main__":
    rag_system = DeepSeekRAGSystem()
    rag_system.chat_loop()
