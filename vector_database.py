from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# 1. 加载PDF文档（批量处理目录中的pdf文件）
loader = DirectoryLoader(
    path="./doc",  # 替换为PDF文件目录
    glob="**/*.pdf",                # 匹配所有PDF文件
    loader_cls=PyPDFLoader,         # 使用PDF专用加载器
    show_progress=True
)
documents = loader.load()

# 2. 分割文本（适配PDF文档特性）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # 根据PDF内容密度调整
    chunk_overlap=200,   # 确保上下文连贯性
    separators=["\n\n", "\n", "。", " "],  # 中文分段优化
    length_function=len
)
texts = text_splitter.split_documents(documents)
print(f"已处理 {len(texts)} 个文本块（来自 {len(documents)} 个PDF页面）")

# 3. 初始化嵌入模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # 推荐嵌入模型
    base_url="http://localhost:11434"
)

# 4. 创建向量数据库
vector_db = FAISS.from_documents(
    documents=texts,
    embedding=embeddings
)
vector_db.save_local("pdf_vectorstore")  # 保存向量数据库

# 5. 构建问答系统
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama2"),  # 问答模型
    chain_type="stuff",
    retriever=vector_db.as_retriever(
        search_type="mmr",       # 最大边际相关性搜索
        search_kwargs={"k": 4}   # 检索4个相关片段
    ),
    return_source_documents=True
)

# 6. 执行问答
query = "财资云项目单笔转账的具体流程？"
result = qa_chain.invoke({"query": query})

# 打印结构化结果
print("="*50)
print(f"问题：{query}")
print("="*50)
print("答案：\n", result["result"])
print("\n来源参考：")
for i, doc in enumerate(result["source_documents"][:2], 1):
    print(f"[参考{i}] PDF页码：{doc.metadata['page']}\n内容片段：{doc.page_content[:200]}...\n")
