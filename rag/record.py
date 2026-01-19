import os
import bs4
import asyncio

# ========== 环境配置 ==========
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ["DASHSCOPE_API_KEY"] = "sk-4431e38c85224bf3aee564da442729c6"

# ========== 導入依賴 ==========
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage,SystemMessage
from supabase.client import Client,create_client

# ========== 初始化模型 ==========
llm = ChatOpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name= "qwen-plus",
    temperature=0.7,
    max_tokens=1000
)

# ========== 加载文档 + 构建检索器 ==========
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 文本切割
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# 存储切割后文本并保存在向量数据库Chroma中,persist_directory指持久化目录
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
# vectorstore.add_texts(
#     texts=["i worked at kensho"],
#     metadatas=[{"namespace": "harrison"}]  # 用 metadata 替代 namespace
# )
# vectorstore.add_texts(
#     texts=["i worked at facebook"],
#     metadatas=[{"namespace": "ankush"}]
# )
# result1 = vectorstore.as_retriever(search_kwargs={"k": 1,"filter": {"namespace": "ankush"}})._get_relevant_documents("where did i work?",run_manager=None)
# result2 = vectorstore.as_retriever(search_kwargs={"k": 1,"filter": {"namespace": "harrison"}})._get_relevant_documents("where did i work?",run_manager=None)
# print(result1)
# print(result2)

# 根据是否有历史聊天记录来传输，否则返回原问题
def contextualized_question(input: dict) -> str:
    if input.get("chat_history"):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question,"
            "which might reference context in the that history,"
            "formulate a standalone question which can be understood,"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it  if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = contextualize_q_prompt | llm | StrOutputParser()
        return chain.stream(input)
    else:
        return input["input"]

# 构建本次问题的上下文
system_prompt = (
    "你是一个智能助手，能调用工具回答问题。\n"
    "你必须遵守以下规则：\n"
    "- 所有回答必须使用简体中文。\n"
    "- 如果问题涉及任务分解（Task Decomposition）、自主智能体（Autonomous Agents）等主题，"
    "请优先调用 blog_post_retriever 工具获取信息。\n"
    "- 回答要简洁，最多三句话。\n"
    "- 列出所有用來回答問題的信息來源(作者+年份+網頁鏈接),用中文回答\n"
    "- 如果不知道答案，就说“我不知道”。\n\n"
    "{tools}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 參數從左到右：向量數據庫，工具名稱，工具描述
def tool_use():
    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        description="""
        专门用于检索Lilian Weng博客《Autonomous Agents》的内容，必须用于回答以下问题：
        - Task Decomposition（任务分解）的定义、常见方法
        - LLM自主智能体（autonomous agent）的相关知识
        - 智能体的任务分解、内存管理、工具使用逻辑
        调用此工具时，传入用户的原始问题作为参数。
        """,
    )
    tools = [tool]
    return tools

# 代理執行器，只要大模型檢測到有代理相關的，就會調用對應的工具庫裏的工具
# langgraph.checkpoint.memory比起ChatMessageHistory可以存儲更多更全的信息，比如
# (對話歷史、工具調用記錄、agent執行節點的位置、思考過程，并且支持斷點續跑)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
contextualize_q_llm = llm.with_config(tags=["contextualize_q_llm"])
agent_executor = create_agent(contextualize_q_llm,tools=tool_use(), checkpointer=memory)
async def async_test():
    query = "JIT编译器在JVM中吗? "
    config = {"configurable": {"thread_id": "abc123"}}
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content=query),SystemMessage(content=f"{qa_prompt}\n\n")]}, config=config,
        include_types=["llm_stream","tool","agent","chain","llm"]
    ):
        event_type = event["event"]
        if event_type == "on_chain_start" and event.get("name") == "LangGraph":
            print(f"【问题】：{event['data']['input']['messages'][0].content}")
        elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
            print(f"【LLM回答】: {event['data']['output']['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(async_test())