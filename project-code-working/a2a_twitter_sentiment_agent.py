from a2a.agent import Agent
from a2a.schema import ToolDefinition
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class TwitterAgent(Agent):
    def __init__(self):
        super().__init__("twitter_sentiment")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            corpus = pd.read_csv("iphone.csv").iloc[:, -1].dropna().astype(str).tolist()
            self.corpus = corpus
            self.embeddings = self.embedder.encode(corpus[:50], convert_to_tensor=True)
            print(f"[{self.name}] Loaded {len(self.corpus)} iPhone entries", flush=True)
        except Exception as e:
            print(f"[{self.name}] Failed to load corpus: {e}", flush=True)
            self.corpus = []
            self.embeddings = None

    async def onInit(self):
        return [ToolDefinition(name="analyze_prompt", parameters={"text": {"type": "string"}})]

    async def analyze_prompt(self, text: str):
        print(f" twitter Gangadhar [{self.name}] analyze_prompt called!", flush=True)
        sentiment_raw = self.sentiment_model(text[:512])[0]
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        label = label_map.get(sentiment_raw["label"], sentiment_raw["label"])

        if self.embeddings is not None and self.corpus:
            query_emb = self.embedder.encode(text, convert_to_tensor=True)
            match = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
            top = match.argmax().item()
            similar = self.corpus[top]
        else:
            similar = "N/A"

        return {
            "text": f"Most similar review: \"{similar}\"",
            "sentiment": label
        }

if __name__ == "__main__":
    import asyncio
    asyncio.run(TwitterAgent().run())

