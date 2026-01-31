from a2a.agent import Agent
from a2a.schema import ToolDefinition
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import asyncio
import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

test_samples = [
    {"prompt": "The iPhone battery drains too fast.", "true_sentiment": "NEGATIVE"},
    {"prompt": "The new iPhone design is stunning!", "true_sentiment": "POSITIVE"},
    {"prompt": "Twitter keeps crashing on my phone.", "true_sentiment": "NEGATIVE"},
    {"prompt": "I had an amazing time using Twitter Spaces.", "true_sentiment": "POSITIVE"},
    {"prompt": "Why are people unhappy with the new iPhone?", "true_sentiment": "NEGATIVE"},
    {"prompt": "Twitter is full of trolls lately.", "true_sentiment": "NEGATIVE"},
    {"prompt": "Are iPhone users complaining on Twitter?", "true_sentiment": "NEGATIVE"},
    {"prompt": "I saw a tweet saying iPhones overheat easily.", "true_sentiment": "NEGATIVE"},
    {"prompt": "I really enjoy using Twitter every day.", "true_sentiment": "POSITIVE"},
    {"prompt": "iPhones are overpriced but still worth it.", "true_sentiment": "POSITIVE"}
]

# ------------------- Metrics Calculation ------------------- #

async def evaluate_metrics(test_samples, agent):
    y_true = []
    y_pred = []
    y_scores = []

    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    class_labels = list(label_map.keys())

    for sample in test_samples:
        result = await agent.analyze_prompt(sample["prompt"])
        print(f"→ Response: {result}")
        sentiment_field = result["sentiment"] if isinstance(result, dict) and "sentiment" in result else "UNKNOWN (0.0)"
        label = sentiment_field.split()[0]
        try:
            score = float(sentiment_field.split("(")[-1].rstrip(")"))
        except:
            score = 0.0

        y_true.append(sample["true_sentiment"])
        y_pred.append(label)

        prob_vector = [0.0, 0.0, 0.0]
        if label in label_map:
            prob_vector[label_map[label]] = score
        y_scores.append(prob_vector)

    print("\n--- Sentiment Classification Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="weighted", zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if len(set(y_true)) < 2:
        print("ROC-AUC Score: Skipped (only one class in y_true)")
    else:
        y_true_bin = label_binarize(y_true, classes=class_labels)
        print("ROC-AUC Score:", roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr"))


class IphoneSentiment(Agent):
    def __init__(self):
        super().__init__("iphone_sentiment")
        
        # Updated specific paths for the iPhone model
        sentiment_path = "./models/distilbert_sentiment"
        embedder_path = "./models/minilm_embedder_iphone"
        
        os.makedirs("./models", exist_ok=True)

        # 1. Sentiment Model Logic
        if os.path.exists(sentiment_path):
            print(f"[{self.name}] Loading local iPhone sentiment model...")
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model=sentiment_path, 
                tokenizer=sentiment_path
            )
        else:
            print(f"[{self.name}] Downloading iPhone sentiment model...")
            self.sentiment_model = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.sentiment_model.model.save_pretrained(sentiment_path)
            self.sentiment_model.tokenizer.save_pretrained(sentiment_path)

        # 2. Embedder Logic (Using the new _iphone path)
        if os.path.exists(embedder_path):
            print(f"[{self.name}] Loading local iPhone embedder...")
            self.embedder = SentenceTransformer(embedder_path)
        else:
            print(f"[{self.name}] Downloading iPhone embedder...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedder.save(embedder_path)
            print(f"[{self.name}] Embedder saved to {embedder_path}")

        # 3. Data Loading
        try:
            df = pd.read_csv("iphone.csv")
            corpus = df.iloc[:, -1].dropna().astype(str).tolist()
            self.corpus = corpus
            self.embeddings = self.embedder.encode(corpus[:50], convert_to_tensor=True)
            print(f"[{self.name}] Loaded {len(self.corpus)} entries", flush=True)
        except Exception as e:
            print(f"[{self.name}] Failed to load corpus: {e}", flush=True)
            self.corpus = []
            self.embeddings = None
    async def onInit(self):
        return [ToolDefinition(name="analyze_prompt", parameters={"text": {"type": "string"}})]

    async def analyze_prompt1(self, text: str):
        sentiment_raw = self.sentiment_model(text[:512])[0]
        label = sentiment_raw["label"]

        if label == "LABEL_0":
            label = "NEGATIVE"
        elif label == "LABEL_1":
            label = "NEUTRAL"
        elif label == "LABEL_2":
            label = "POSITIVE"

        if self.embeddings is not None:
            query_emb = self.embedder.encode(text, convert_to_tensor=True)
            match = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
            top = match.argmax().item()
            similar = self.corpus[top]
        else:
            similar = "N/A"

        return f"[✓ {self.name}] {label}: {text}"

    async def analyze_prompt(self, text: str):
        # Sentiment Analysis
        sentiment = self.sentiment_model(text[:512])[0]  # handles long text
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        label = label_map.get(sentiment["label"], sentiment["label"])
        score = round(sentiment["score"], 3)

        # Optional: Match with similar corpus entries using embedding
        if self.embeddings is not None and self.corpus:
            query_emb = self.embedder.encode(text, convert_to_tensor=True)
            hits = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
            top_idx = int(hits.argmax())
            example = self.corpus[top_idx]
        else:
            example = "No similar review found."

        return {
            "text": f"Most similar user review: \"{example}\"",
            "sentiment": f"{label} ({score})"
        }

if __name__ == "__main__":
    import asyncio
    from a2a_iphone_sentiment_agent import IphoneSentiment  # or from current module

    test_samples = [
        {"prompt": "The iPhone battery drains too fast.", "true_sentiment": "NEGATIVE"},
        {"prompt": "The new iPhone design is stunning!", "true_sentiment": "POSITIVE"},
        {"prompt": "Why are people unhappy with the new iPhone?", "true_sentiment": "NEGATIVE"},
        {"prompt": "iPhones are overpriced but still worth it.", "true_sentiment": "POSITIVE"},
        {"prompt": "I saw a tweet saying iPhones overheat easily.", "true_sentiment": "NEGATIVE"},
    ]

    async def main():
        agent = IphoneSentiment()
        await evaluate_metrics(test_samples, agent)

    asyncio.run(main())

