from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# 定义提示模板
prompt = ChatPromptTemplate.from_template(
    "你是一个资深科学家，请用简单语言解释以下概念：{concept}"
)

# 初始化模型和链
model = Ollama(model="deepseek-r1:8b")
chain = prompt | model

# 运行链
response = chain.invoke({"concept": "量子纠缠"})
print(response)
