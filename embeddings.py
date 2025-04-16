import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./doc/test.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(docs)

from langchain_community.embeddings import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(
    model='llama2',
    base_url="http://localhost:11434",
)

texts = [doc.page_content for doc in docs]
embeddings = embeddings_model.embed_documents(texts)
print(len(embeddings))
print(len(embeddings[0]))
