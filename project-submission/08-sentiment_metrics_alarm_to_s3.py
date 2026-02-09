import json
import time
import math
import random
import boto3
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
CSV_PATH = "iphone.csv"
ENDPOINT_NAME = "sentiment-baseline-endpoint"

BUCKET_NAME = "my-sentiment-monitoring-bucket"
S3_PREFIX = "alarms/"

SAMPLE_SIZE = 600
POSITIVE_RATIO = 0.4   # 40% positive-leaning, 60% random
MAX_TEXT_LENGTH = 512

CONFIDENCE_THRESHOLD = 0.60
NEGATIVE_RATE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.40

random.seed(42)  # reproducible sampling (optional)

runtime = boto3.client("sagemaker-runtime")
s3 = boto3.client("s3")

# =========================
# LOAD REVIEWS
# =========================
def load_reviews(csv_path):
    df = pd.read_csv(csv_path)

    if not {"reviewTitle", "reviewDescription"}.issubset(df.columns):
        raise ValueError("CSV must contain 'reviewTitle' and 'reviewDescription'")

    df["text"] = (
        df["reviewTitle"].fillna("") + ". " +
        df["reviewDescription"].fillna("")
    )

    reviews = df["text"].dropna().tolist()
    reviews = [r.strip() for r in reviews if r.strip()]

    return reviews

# =========================
# MIXED SAMPLING
# =========================
def sample_mixed_reviews(reviews, total=20, positive_ratio=0.4):
    positive_keywords = [
        "good", "great", "excellent", "amazing",
        "love", "best", "perfect", "awesome"
    ]

    positives = [
        r for r in reviews
        if any(k in r.lower() for k in positive_keywords)
    ]

    num_positive = min(int(total * positive_ratio), len(positives))
    num_random = total - num_positive

    sampled = []

    if num_positive > 0:
        sampled.extend(random.sample(positives, num_positive))

    remaining = list(set(reviews) - set(sampled))
    sampled.extend(random.sample(remaining, min(num_random, len(remaining))))

    random.shuffle(sampled)
    return sampled

# =========================
# INFERENCE (SAFE)
# =========================
def predict_sentiment(text):
    payload = json.dumps({
        "inputs": text,
        "parameters": {
            "truncation": True,
            "max_length": MAX_TEXT_LENGTH
        }
    })

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read())[0]

    label = "positive" if result["label"] == "LABEL_1" else "negative"

    return {
        "text": text[:200] + "..." if len(text) > 200 else text,
        "label": label,
        "score": result["score"]
    }

# =========================
# METRICS
# =========================
def entropy(p):
    return -sum(x * math.log(x + 1e-9) for x in p)

def compute_metrics(results):
    sentiments = []
    confidences = []
    lengths = []
    entropies = []

    for r in results:
        sentiments.append(r["label"])
        confidences.append(r["score"])
        lengths.append(len(r["text"].split()))
        entropies.append(entropy([r["score"], 1 - r["score"]]))

    total = len(results)

    return {
        "TotalSamples": total,
        "PositiveRate": sentiments.count("positive") / total,
        "NegativeRate": sentiments.count("negative") / total,
        "LowConfidenceRate": sum(c < CONFIDENCE_THRESHOLD for c in confidences) / total,
        "AvgConfidence": float(np.mean(confidences)),
        "ConfidenceP90": float(np.percentile(confidences, 90)),
        "AvgTextLength": float(np.mean(lengths)),
        "P95TextLength": float(np.percentile(lengths, 95)),
        "AvgEntropy": float(np.mean(entropies)),
        "Timestamp": datetime.utcnow().isoformat()
    }

# =========================
# ALARMS
# =========================
def evaluate_alarms(metrics):
    alarms = []

    if metrics["NegativeRate"] > NEGATIVE_RATE_THRESHOLD:
        alarms.append("🚨 High Negative Rate")

    if metrics["LowConfidenceRate"] > LOW_CONFIDENCE_THRESHOLD:
        alarms.append("⚠️ Low Confidence Spike")

    if metrics["P95TextLength"] > MAX_TEXT_LENGTH:
        alarms.append("🚨 Input Text Length Drift")

    if metrics["AvgEntropy"] > 0.65:
        alarms.append("⚠️ High Prediction Uncertainty")

    return alarms

# =========================
# S3 HELPERS
# =========================
def ensure_bucket(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"✅ S3 bucket exists: {bucket}")
    except:
        print(f"🪣 Creating S3 bucket: {bucket}")
        s3.create_bucket(Bucket=bucket)

def upload_snapshot(data):
    ts = int(time.time())
    key = f"{S3_PREFIX}alarm_snapshot_{ts}.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )

    return key

def read_snapshot_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_bucket(BUCKET_NAME)

    reviews = load_reviews(CSV_PATH)

    sampled_reviews = sample_mixed_reviews(
        reviews,
        total=SAMPLE_SIZE,
        positive_ratio=POSITIVE_RATIO
    )

    print(f"\n🔍 Running sentiment inference on {len(sampled_reviews)} MIXED reviews...\n")

    results = []
    for i, review in enumerate(sampled_reviews, start=1):
        res = predict_sentiment(review)
        results.append(res)
        print(f"{i}. {res['label']} ({res['score']:.2f})")

    metrics = compute_metrics(results)
    alarms = evaluate_alarms(metrics)

    snapshot = {
        "metrics": metrics,
        "alarms": alarms
    }

    s3_key = upload_snapshot(snapshot)

    print("\n📊 METRICS")
    print(json.dumps(metrics, indent=2))

    print("\n🚨 ALARMS")
    print(alarms if alarms else "No alarms triggered")

    print(f"\n📦 Snapshot uploaded to s3://{BUCKET_NAME}/{s3_key}")

    snapshot_from_s3 = read_snapshot_from_s3(BUCKET_NAME, s3_key)

    print("\n📥 SNAPSHOT READ BACK FROM S3")
    print(json.dumps(snapshot_from_s3, indent=2))

