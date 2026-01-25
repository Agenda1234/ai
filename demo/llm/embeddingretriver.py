import os
from vectorstore import VectorStore
from langchain_community.embeddings import DashScopeEmbeddings

os.environ["DASHSCOPE_API_KEY"] = "sk-4431e38c85224bf3aee564da442729c6"

class EmbeddingRetriever:
    def __init__(self, model):
        self.embeddingModel = model
        self.vectorStore = VectorStore()
        self.key = os.environ["DASHSCOPE_API_KEY"]

    async def embedDocument(self, text):
        doc_emb = await self.embed(text)
        self.vectorStore.add(doc_emb, text)
        return doc_emb
    
    async def embedQuery(self, query):
        return await self.embed(query)
    
    async def embed(self, text):
        embeddings = DashScopeEmbeddings(model=self.embeddingModel, dashscope_api_key=self.key)
        vector = await embeddings.aembed_documents(texts=text)
        return vector
    
    async def retrieve(self, query: str, topk: int = 3):
        query_emb = await self.embedQuery(query=query)
        return self.vectorStore.search(query_emb, topk)