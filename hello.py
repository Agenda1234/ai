import os
from openai import OpenAI

# ====================== 1. 配置阿里通义千问API ======================
DASHSCOPE_API_KEY = "sk-111"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 初始化千问大模型客户端（仅未命中时调用）
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ====================== 2. 核心函数：优先读向量库，命中直接返回，未命中调用大模型 ======================
def retrieve_or_answer(query: str, vectorstore, similarity_threshold: float = 0.5):
    """
    核心逻辑：
    1. 从向量库检索，命中（相似度≥阈值）→ 直接返回检索结果；
    2. 未命中 → 调用大模型用通用知识回答。
    :param query: 用户查询问题
    :param vectorstore: Chroma向量库对象
    :param similarity_threshold: 相似度阈值（0-1）
    :return: 最终回答
    """
    # 步骤1：从向量库检索（带相似度分数）
    results = vectorstore.similarity_search_with_score(query=query, k=3)
    
    # 步骤2：过滤有效结果（相似度≥阈值）
    valid_results = [(doc, score) for doc, score in results if score >= similarity_threshold]
    
    # 分支1：命中 → 直接返回检索到的内容
    if valid_results:
        # 拼接所有有效检索结果
        retrieved_content = "\n\n".join([
            f"【相关内容 {i+1}（相似度：{score:.2f}）】\n{doc.page_content}"
            for i, (doc, score) in enumerate(valid_results)
        ])
        return f"✅ 从文档中找到相关内容：\n{retrieved_content}"
    
    # 分支2：未命中 → 调用大模型回答
    else:
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是专业的助手，用通用知识回答用户问题。"},
                    {"role": "user", "content": query},
                ]
            )
            answer = completion.choices[0].message.content
            return f"📝 未从文档中找到相关内容，以下是通用回答：\n{answer}"
        except Exception as e:
            return f"❌ 大模型调用失败：{str(e)}"

# ====================== 3. 加载PDF+拆分+向量化+存入向量库 ======================
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.dashscope import DashScopeEmbeddings
from langchain.vectorstores import Chroma

# 加载PDF
loader = PyPDFLoader("agendadu.pdf")
pages = loader.load_and_split()
print(f"✅ PDF加载完成，总页数: {len(pages)}")

# 拆分文本
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
docs = text_splitter.split_documents(pages)
print(f"✅ 文本拆分完成，总段落数: {len(docs)}")

# 初始化阿里嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=DASHSCOPE_API_KEY)

# 存入向量库（持久化）
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="agenda",
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("✅ 文本向量化完成，已存入Chroma向量数据库")

# ====================== 4. 测试：命中返回检索结果，未命中调用大模型 ======================
# 测试1：命中的查询（PDF里有相关内容）→ 直接返回检索结果
query1 = "他在哪家企业工作过"
print(f"\n🔍 查询1：{query1}")
answer1 = retrieve_or_answer(query1, vectorstore, similarity_threshold=0.5)
print(answer1)

# 测试2：未命中的查询（PDF里无相关内容）→ 调用大模型回答
query2 = "2025年英雄联盟S15冠军是谁"
print(f"\n🔍 查询2：{query2}")
answer2 = retrieve_or_answer(query2, vectorstore, similarity_threshold=0.5)
print(answer2)
