from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self,name,args,command):
        self.name = name
        self.command = command
        self.args = args
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def init(self):
        await self.connect_to_server()
    
    async def close(self):
        try:
            self.session = None
            try:
                if hasattr(self, 'exit_stack'):
                    await self.exit_stack.aclose()
            except:
                pass
        except:
            pass
    
    def get_tools(self):
        return self.tools
    
    async def call_tool(self, name:str, params: dict):
        return await self.session.call_tool(name=name, arguments=params)

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command=self.command,
            args = self.args
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools

        self.tools = []

        for tool in tools:
            tool_dict = {
                "name": tool.name if hasattr(tool, 'name') else  str(tool),
                "description": tool.description if hasattr(tool, 'description') else "",
                "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else (tool.inputSchema if hasattr(tool, 'inputSchema') else {})
            }
            self.tools.append(tool_dict)
        print("\nConnected to server with tools:", [tool.name for tool in tools])
    