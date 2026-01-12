import os
from openai import OpenAI

# 配置千问API
DASHSCOPE_API_KEY = ""
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 核心检索函数（新增阈值展示逻辑）
def retrieve_or_answer(query: str, vectorstore, distance_threshold: float = 1.0):
    """
    核心逻辑：
    1. Chroma返回的是余弦距离（0-2），距离≤阈值才视为命中（距离越小相似度越高）；
    2. 未命中时展示当前使用的阈值，方便排查；
    """
    results = vectorstore.similarity_search_with_score(query=query, k=3)
    valid_results = [(doc, score) for doc, score in results if score <= distance_threshold]
    
    if valid_results:
        retrieved_content = "\n\n".join([
            f"【相关内容 {i+1}（距离：{score:.2f}）】\n{doc.page_content}"
            for i, (doc, score) in enumerate(valid_results)
        ])
        return f"✅ 从文档中找到相关内容：\n{retrieved_content}"
    else:
        # 未命中时，补充展示阈值和检索到的最低距离（便于调试）
        # 获取所有检索结果的距离，展示最接近的那个
        if results:
            min_distance = min([score for _, score in results])
            hint = f"（当前距离阈值：{distance_threshold}，检索到的最小距离：{min_distance:.2f}）"
        else:
            hint = f"（当前距离阈值：{distance_threshold}，未检索到任何内容）"
        
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是专业的助手，用通用知识回答用户问题。"},
                    {"role": "user", "content": query},
                ]
            )
            answer = completion.choices[0].message.content
            return f"📝 未从文档中找到相关内容 {hint}，以下是通用回答：\n{answer}"
        except Exception as e:
            return f"❌ 大模型调用失败 {hint}：{str(e)}"

# 加载PDF+拆分（新版导入路径）
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 加载PDF
loader = PyPDFLoader("agendadu.pdf")
raw_pages = loader.load()
print(f"✅ PDF加载完成，原始页数: {len(raw_pages)}")

# 标准化Document
pages = []
for page in raw_pages:
    content = page.page_content.strip() if hasattr(page, "page_content") else ""
    if content:
        pages.append(Document(page_content=content, metadata=page.metadata if hasattr(page, "metadata") else {}))

# 拆分文本
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
docs = text_splitter.split_documents(pages)

# 去重
unique_docs = []
seen_content = set()
for doc in docs:
    content = doc.page_content.strip()
    if content not in seen_content:
        seen_content.add(content)
        unique_docs.append(doc)
docs = unique_docs
print(f"✅ 文本拆分+去重完成，总段落数: {len(docs)}")

# 向量化+存储
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="agenda",
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("✅ 文本向量化完成，已存入Chroma向量数据库")

# 测试
query1 = "他哪年在哪家企业工作过"
print(f"\n🔍 查询1：{query1}")
print(retrieve_or_answer(query1, vectorstore, distance_threshold=1.1))

query2 = "你是谁"
print(f"\n🔍 查询2：{query2}")
<<<<<<< HEAD
print(retrieve_or_answer(query2, vectorstore, distance_threshold=1.0))
=======
answer2 = retrieve_or_answer(query2, vectorstore, similarity_threshold=0.5)
print(answer2)
>>>>>>> 34d4026955bdd258d2df7cd5e8960e6d2c0b5c10
