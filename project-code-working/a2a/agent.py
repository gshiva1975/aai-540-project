
import asyncio

class Agent:
    def __init__(self, name="agent"):
        self.name = name
        print(f"[A2A Agent: {self.name}] Initialized")

    async def onInit(self):
        return []

    async def onMessage(self, msg):
        print(f"[{self.name}] Received message: {msg.content}")
        await self.send(self.respond(msg, "Default response"))

    async def send(self, response):
        print(f"[{self.name}] Sending response: {response.content}")

    async def call_tool(self, tool_call):
        print(f"[{self.name}] Calling tool: {tool_call.name}", flush=True)
        fn = getattr(self, tool_call.name, None)
        if fn is None:
            raise Exception(f"Tool '{tool_call.name}' not implemented in {self.name}")
        result = await fn(**tool_call.arguments)
        return type("ToolResult", (), {"output": result})()

    def respond(self, msg, content):
        return type("Response", (), {"content": content})

    async def run(self):
        print(f"[{self.name}] Agent running... (stubbed)")
        while True:
            await asyncio.sleep(3600)

    async def run_forever(self):
        await self.onInit()
        print(f"[{self.name}] Agent running... (listening mode)", flush=True)
        while True:
            await asyncio.sleep(3600)
