
import asyncio
from a2a.agent import Agent
from a2a.schema import ToolCall, ToolResult
from a2a.message import Message
from a2a.routing import ZeroShotRouter
from a2a.schema import ToolDefinition

from a2a_iphone_sentiment_agent import IPhoneAgent as IPhoneAgent
from a2a_twitter_sentiment_agent import TwitterAgent as TwitterAgentAgent

class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__("coordinator_agent")
        self.router = ZeroShotRouter()
        self.agents = {
            "iphone_sentiment": IPhoneAgent(),
            "twitter_sentiment": TwitterAgentAgent()
        }
        self.tools = {
            "iphone_sentiment": "iPhone-related issues or praise",
            "twitter_sentiment": "Twitter-related experiences or comments"
        }

    async def onInit(self):
        return []

    async def onMessage(self, msg: Message):
        prompt = msg.content

        tool, label, score = self.router.route(prompt, self.tools)
        print(f"[Routing] '{prompt}' → {tool} (label='{label}', score={score:.2f})", flush=True)

        tool_call = ToolCall(tool=tool, name="analyze_prompt", arguments={"text": prompt})
        print(f"[coordinator_agent] Calling tool: {tool_call.name} on agent: {tool_call.tool}", flush=True)

        output: ToolResult = await self.agents[tool_call.tool].call_tool(tool_call)
        print(f"[Gangadhar main] {output.output}", flush=True)

        sentiment = output.output.get("sentiment")
        text = output.output.get("text")

        await self.send(msg.respond(f"[✓ {tool}] {sentiment}: {text}"))

if __name__ == "__main__":
    agent = CoordinatorAgent()
    asyncio.run(agent.run())
