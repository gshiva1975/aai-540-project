
from a2a_main import CoordinatorAgent
from a2a.message import Message
import asyncio

class A2AClient:
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self._initialized = False

    async def _init_if_needed(self):
        if not self._initialized:
            await self.coordinator.onInit()
            self._initialized = True

    async def send(self, prompt):
        await self._init_if_needed()

        class FakeMessage:
            def __init__(self, content):
                self.content = content
            def respond(self, content):
                return FakeMessage(content)

        msg = FakeMessage(prompt)
        await self.coordinator.onMessage(msg)
        return FakeMessage(f"[✓ coordinator_agent] Processed: {prompt}")
