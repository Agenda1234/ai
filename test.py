import os
import logging
from pypdf import PdfReader
from typing import List, Set
from langchain_core.documents import Document

# LangChain 1.x核心导入
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

# 关键替换：用ChatOpenAI对接阿里云兼容接口（无需langchain-dashscope）
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 文本拆分、向量存储相关
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== 配置阿里云API密钥 =====================
DASHSCOPE_API_KEY = ""
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

def extract_text_with_documents(pdf) -> List[Document]:
    """读取PDF并生成带页码的Document对象"""
    documents = []
    for page_number, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            doc = Document(
                page_content=page_text,
                metadata={"page_number": page_number}
            )
            documents.append(doc)
        else:
            logging.warning(f"Page {page_number} has no extractable text.")
    return documents

def process_text_with_splitter(documents: List[Document]) -> FAISS:
    """拆分文本并构建带页码元数据的FAISS向量库"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Text splitting completed, total chunks: {len(split_docs)}")

    # 初始化阿里云Embedding
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    knowledge_base = FAISS.from_documents(split_docs, embeddings)
    print("✅ Vector store created using FAISS (with page metadata).")

    return knowledge_base

def test(query: str, knowledge_base: FAISS):
    """测试QA链（兼容langchain-core 1.x）"""
    if not query:
        logging.warning("查询内容不能为空！")
        return

    # 关键：用ChatOpenAI对接阿里云通义千问兼容接口（无需langchain-dashscope）
    llm = ChatOpenAI(
        model="qwen-turbo",  # 通义千问模型名
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=1024
    )

    # 定义Prompt模板
    prompt = ChatPromptTemplate.from_template("""
    请根据以下上下文回答问题，仅使用上下文信息，不要编造内容：
    上下文：{context}
    问题：{input}
    """)

    # 构建文档合并链
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # 构建检索+回答链
    retrieval_chain = create_retrieval_chain(
        retriever=knowledge_base.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=doc_chain
    )

    # 执行问答链+成本统计
    with get_openai_callback() as cost:
        response = retrieval_chain.invoke({"input": query})
        
        # 输出结果
        print(f"\n✅ 查询已处理。成本统计：{cost}")
        print(f"🤖 回答：{response['answer']}")
        print("📄 来源页码：")
        
        # 提取唯一页码
        unique_pages: Set[int] = set()
        for doc in response["context"]:
            source_page = doc.metadata.get("page_number", "未知")
            if source_page not in unique_pages:
                unique_pages.add(source_page)
                print(f"  - 页码：{source_page} | 片段预览：{doc.page_content[:100]}...")

if __name__ == "__main__":
    try:
        # 1. 读取PDF
        pdf_reader = PdfReader("agendadu.pdf")
        # 2. 提取带页码的文本
        documents = extract_text_with_documents(pdf_reader)
        print(f"✅ 提取到的有效页面数: {len(documents)}")
        # 3. 构建向量存储
        knowledge_base = process_text_with_splitter(documents)
        print("✅ 向量库构建完成！")

        # 4. 测试检索
        query = "PDF中，杜艺铖的教育经历是哪个大学？"
        relevant_docs = knowledge_base.similarity_search(query, k=2)
        print(f"\n🔍 检索到{len(relevant_docs)}条相关片段：")
        for i, doc in enumerate(relevant_docs):
            print(f"  片段{i+1}（页码{doc.metadata['page_number']}）：{doc.page_content[:100]}...")

        # 5. 运行QA链测试
        test(query, knowledge_base)
            
    except Exception as e:
        logging.error(f"❌ 执行失败：{str(e)}")
        import traceback
        traceback.print_exc()