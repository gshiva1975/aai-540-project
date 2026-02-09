import re
import json
import boto3
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = "iphone.csv"
ENDPOINT_NAME = "sentiment-baseline-endpoint"
REGION = "us-east-1"

MAX_CHARS = 1500   # ~400 tokens safe limit

# =====================================================
# CLIENT
# =====================================================
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# =====================================================
# LABEL NORMALIZATION
# =====================================================
def normalize_label(label: str) -> str:
    if label.upper() == "LABEL_1":
        return "POSITIVE"
    if label.upper() == "LABEL_0":
        return "NEGATIVE"
    return label

# =====================================================
# PII ANONYMIZATION
# =====================================================
def anonymize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    patterns = {
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+": "<EMAIL>",
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b": "<PHONE>",
        r"\b\d{5,}\b": "<ID>",
        r"\b[A-Z][a-z]{2,}\b": "<NAME>",
    }

    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    return text

# =====================================================
# SAFE TRUNCATION (CRITICAL FIX)
# =====================================================
def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# =====================================================
# LOAD REVIEWS (FIXED COLUMNS)
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
        anonymized = anonymize_text(combined)
        truncated = truncate_text(anonymized)
        reviews.append(truncated)

    return reviews

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
        "score": round(result["score"], 4),
    }

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print(" Loading iPhone reviews...")
    reviews = load_reviews(CSV_PATH)

    print(f"\n Running sentiment inference on {len(reviews)} reviews...\n")

    for i, review in enumerate(reviews[:20], start=1):
        result = predict_sentiment(review)

        print(f"Review #{i}")
        print(f"Text      : {review[:200]}{'...' if len(review) > 200 else ''}")
        print(f"Sentiment : {result['label']} ({result['score']})")
        print("-" * 70)

