import json
import random
import boto3
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = "iphone.csv"
ENDPOINT_NAME = "sentiment-baseline-endpoint"
REGION = "us-east-1"

MAX_CHARS = 1500      # safe limit (~400 tokens)
SAMPLE_SIZE = 20      # randomized sample size

# =====================================================
# AWS CLIENT
# =====================================================
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# =====================================================
# LABEL NORMALIZATION (FIXED)
# =====================================================
def normalize_label(label: str) -> str:
    mapping = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
    }
    return mapping.get(label.upper(), label)

# =====================================================
# SAFE TRUNCATION
# =====================================================
def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    if not isinstance(text, str):
        return ""
    return text[:max_chars]

# =====================================================
# LOAD & RANDOMIZE REVIEWS (FIXED)
# =====================================================
def load_reviews(csv_path: str):
    df = pd.read_csv(csv_path)

    required_cols = {"reviewTitle", "reviewDescription"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, found {set(df.columns)}"
        )

    reviews = []

    for _, row in df.iterrows():
        combined = f"{row['reviewTitle']} {row['reviewDescription']}"
        reviews.append(truncate_text(combined.strip()))

    # Randomize to avoid bias
    return random.sample(reviews, min(SAMPLE_SIZE, len(reviews)))

# =====================================================
# SENTIMENT INFERENCE
# =====================================================
def predict_sentiment(text: str):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": text}),
    )

    result = json.loads(response["Body"].read().decode())[0]

    return {
        "label": normalize_label(result["label"]),
        "score": round(float(result["score"]), 4),
    }

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("📱 Loading iPhone reviews...")
    reviews = load_reviews(CSV_PATH)

    print(f"\n🚀 Running sentiment inference on {len(reviews)} random reviews...\n")

    for i, review in enumerate(reviews, start=1):
        result = predict_sentiment(review)

        print(f"Review #{i}")
        print(f"Text      : {review[:200]}{'...' if len(review) > 200 else ''}")
        print(f"Sentiment : {result['label']} ({result['score']})")
        print("-" * 70)

