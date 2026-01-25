import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage,AIMessage,SystemMessage

os.environ["DASHSCOPE_API_KEY"] = "sk-4431e38c85224bf3aee564da442729c6"
os.environ["BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class ChatOpenAIFromLangChain():
    def __init__(self, model_name:str, tools = [], system_prompt:str = "", context:str = "", chat_history = []):
        api_key = os.environ["DASHSCOPE_API_KEY"]
        base_url = os.environ["BASE_URL"] 
        self.model = model_name
        self.tools = tools
        self.system_prompt = system_prompt
        self.context = context
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=base_url,
            model=self.model,
            streaming=False,
            temperature=0.1)
        self.message = []

    async def chat(self, prompt = None, history_context = "", tool_call_list = []):
        # 清除上一次对话,避免调用工具带来的影响
        self.message.clear()

        print("本次历史会话" + str(history_context))
        # 构建当前会话，历史会话作为聊天记录传入
        full_prompt = f"""
        {self.system_prompt}
        
        对话历史:
        {history_context}

        用户当前问题:
        {prompt}

        本次需要调用的工具:
        {str(tool_call_list)}
        请根据这些工具的调用结果回答用户问题。
        """

        content = ""

        # 构建调用参数——调用
        invoke_kwargs = {"input": [HumanMessage(content=full_prompt)]}
        if self.tools:
            invoke_kwargs["tools"] = self.getToolsDefinition()
            invoke_kwargs["tool_choice"] = "auto"
        response = await self.llm.ainvoke(**invoke_kwargs)

        # 设置回包
        if hasattr(response, "content"):
            content = response.content
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                self.message.append({
                    "role": "assistant", 
                    "content": content, 
                    "tool_calls": [
                        {
                            "id": tool_call["id"], 
                            "type": "function", 
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["args"]
                            }
                        }
                    ]
                })
        if not self.message:
            self.message.append({
                    "role": "assistant", 
                    "content": content, 
                    "tool_calls": None
                })
        return self.message
                
    def getToolsDefinition(self):
        if self.tools:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool['description'],
                        "parameters": tool['inputSchema']
                    }
                } for tool in self.tools
            ]