# 工具类: 天气
from mcptools.weather import GlobalWeatherMCPClient

import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
from agent import Agent
from embeddingretriver import EmbeddingRetriever

embeddingRetriever = None
agent = None

async def init_global_objects():
    """初始化embedding和agent"""
    global embeddingRetriever,agent
    if embeddingRetriever is None or agent is None:
        # 初始化embedding
        emb_model = "text-embedding-v1"
        embeddingRetriever = EmbeddingRetriever(model=emb_model)

        # 初始化Agent
        agent = Agent(model="qwen-plus", mcpClients=[GlobalWeatherMCPClient()], context=[])
        await agent.init()
    print("初始化embedding和agent完成")

async def chat_with_context(input):
    """复用agent,保留对话上下文"""
    if embeddingRetriever is None or agent is None:
        await init_global_objects()
    
    # 上下文初始化
    context = await embeddingRetriever.retrieve(input, 3)
    agent.context = context
    agent.chat_history.append({"role": "user", "content": input})
    
    resp = await agent.invoke(input)

    agent.chat_history.append({"role": "assistant", "content": resp})

    print(f"📝 对话历史（共{len(agent.chat_history)}轮）：{agent.chat_history}")
    print(f"💡 本次回复：{resp}")
    return resp

async def main(input):
    return await chat_with_context(input)
