import numpy as np
from chatopenai import ChatOpenAIFromLangChain

class Agent():
    def __init__(self, model, mcpClients, system_prompt="", context="",chat_history = []) -> None:
        self.mcpClients = mcpClients
        self.model = model
        self.system_prompt = system_prompt
        self.context = context
        self.llm = None
        self.max_tool_calls = 10
        self.chat_history = chat_history
    
    async def init(self):
        for mcp in self.mcpClients:
            await mcp.init()
        
        all_tools = []
        for client in self.mcpClients:
            all_tools.extend(client.get_tools())
        
        self.llm = ChatOpenAIFromLangChain(self.model, system_prompt=self.system_prompt, tools=all_tools, context=self.context, chat_history = self.chat_history)

    async def close(self):
        for mcp in self.mcpClients:
            try:
                await mcp.close()
            except Exception as e:
                print(f"Warning: Error closing Mcp client: {e}")
    
    async def invoke(self, prompt: str):
        print("invoke_chat_history:" + str(self.chat_history))
        if not self.llm:
            raise Exception("Agent not initialized")
        
        response = await self.llm.chat(prompt = prompt,history_context = self.chat_history)
        while True:
            for response in response:
                if isinstance(response, dict) and "tool_calls" in response and response["tool_calls"]:
                    tool_call_list = []
                    for tool_call in response["tool_calls"]:
                        mcp = next(
                            (client for client in self.mcpClients if any(
                                t['name'] == tool_call['function']['name'] for t in client.get_tools()
                            )),
                             None
                        )

                        if mcp:
                            result = await mcp.call_tool(
                                tool_call['function']['name'],
                                tool_call['function']['arguments']
                            )
                            tool_call_list.append({
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call['id']
                            })
                        else:
                            tool_call_list.append({
                                "role": "tool",
                                "content": 'Tool not found',
                                "tool_call_id": tool_call['id']
                            })
                        response = await self.llm.chat(tool_call_list=tool_call_list,history_context = self.chat_history)
                        continue
            await self.close()
            print(response)
            if isinstance(response, list):
                return response[0]['content']
            return response['content']