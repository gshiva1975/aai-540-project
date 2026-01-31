from a2a.agent import Agent
from a2a.schema import ToolDefinition
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class TwitterSentiment(Agent):
    def __init__(self):
        super().__init__("twitter_sentiment")
        
        # Unique paths for Twitter-specific models
        sentiment_path = "./models/twitter_roberta_sentiment"
        embedder_path = "./models/minilm_embedder_twitter"
        
        os.makedirs("./models", exist_ok=True)

        # 1. Twitter Sentiment Model Logic
        if os.path.exists(sentiment_path):
            print(f"[{self.name}] Loading local Twitter sentiment model...")
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model=sentiment_path, 
                tokenizer=sentiment_path
            )
        else:
            print(f"[{self.name}] Downloading Twitter sentiment model (RoBERTa)...")
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )
            # Fix: Save components individually to avoid the modelcard error
            self.sentiment_model.model.save_pretrained(sentiment_path)
            self.sentiment_model.tokenizer.save_pretrained(sentiment_path)
            print(f"[{self.name}] Twitter model saved locally.")

        # 2. Twitter Embedder Logic
        if os.path.exists(embedder_path):
            print(f"[{self.name}] Loading local Twitter embedder...")
            self.embedder = SentenceTransformer(embedder_path)
        else:
            print(f"[{self.name}] Downloading Twitter embedder...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedder.save(embedder_path)
            print(f"[{self.name}] Twitter embedder saved to {embedder_path}")

        # 3. Data Loading
        try:
            # Assuming you might use a different file for twitter, 
            # but sticking to your snippet's "iphone.csv" if that's your source.
            df = pd.read_csv("iphone.csv") 
            corpus = df.iloc[:, -1].dropna().astype(str).tolist()
            self.corpus = corpus
            self.embeddings = self.embedder.encode(corpus[:50], convert_to_tensor=True)
            print(f"[{self.name}] Loaded {len(self.corpus)} Twitter entries", flush=True)
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
    asyncio.run(TwitterSentiment().run())

