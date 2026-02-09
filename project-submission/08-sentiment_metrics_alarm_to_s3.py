import json
import random
import time
import boto3
import pandas as pd
from collections import Counter

# =============================
# CONFIG
# =============================
CSV_PATH = "iphone.csv"
ENDPOINT_NAME = "sentiment-baseline-endpoint"
REGION = "us-east-1"

BUCKET_NAME = "my-sentiment-monitoring-bucket"
S3_PREFIX = "alarms"

SAMPLE_SIZE = 200
MAX_CHARS = 1000
NEG_THRESHOLD = 0.60

# =============================
# AWS CLIENTS
# =============================
runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# =============================
# HELPERS
# =============================
def ensure_bucket(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"✅ S3 bucket exists: {bucket}")
    except:
        print(f"🪣 Creating S3 bucket: {bucket}")
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )

def truncate_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text[:MAX_CHARS]

def map_label(label: str) -> str:
    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }
    return mapping.get(label.upper(), label.lower())

def invoke_endpoint(text: str):
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": truncate_text(text)}),
    )

    result = json.loads(response["Body"].read())[0]

    return {
        "raw_label": result["label"],
        "sentiment": map_label(result["label"]),
        "confidence": round(float(result["score"]), 4),
    }

# =============================
# LOAD & SAMPLE DATA
# =============================
def load_reviews(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"reviewTitle", "reviewDescription"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}")

    records = df[["reviewTitle", "reviewDescription"]].dropna().to_dict("records")
    return random.sample(records, min(SAMPLE_SIZE, len(records)))

# =============================
# MAIN
# =============================
def main():
    ensure_bucket(BUCKET_NAME)

    print("📱 Loading iPhone reviews...")
    reviews = load_reviews(CSV_PATH)

    print(f"\n🚀 Running sentiment inference on {len(reviews)} reviews...\n")

    results = []
    sentiments = []

    idx = 1
    for r in reviews:
        text = f"{r['reviewTitle']}. {r['reviewDescription']}"
        prediction = invoke_endpoint(text)

        sentiments.append(prediction["sentiment"])

        results.append({
            "reviewTitle": r["reviewTitle"],
            "reviewDescription": r["reviewDescription"],
            "prediction": prediction,
        })

        print(f"{idx}. [{prediction['sentiment'].upper()} | {prediction['confidence']}]")
        print(f"   {r['reviewTitle']}")
        idx += 1

    # =============================
    # OPTIONAL SANITY TESTS
    # =============================
    sanity_tests = [
        "Amazing",
        "Excellent product",
        "Best phone of the decade",
        "Waste of money",
    ]

    print("\n🧪 Sanity test predictions:\n")
    for t in sanity_tests:
        prediction = invoke_endpoint(t)
        sentiments.append(prediction["sentiment"])

        results.append({
            "reviewTitle": t,
            "prediction": prediction,
        })

        print(f"[{prediction['sentiment'].upper()} | {prediction['confidence']}] {t}")

    # =============================
    # METRICS
    # =============================
    counts = Counter(sentiments)
    total = sum(counts.values())

    metrics = {
        "total_reviews": total,
        "positive_count": counts.get("positive", 0),
        "neutral_count": counts.get("neutral", 0),
        "negative_count": counts.get("negative", 0),
        "negative_ratio": round(counts.get("negative", 0) / total, 3),
        "timestamp": int(time.time()),
    }

    alarm_triggered = metrics["negative_ratio"] >= NEG_THRESHOLD

    snapshot = {
        "metrics": metrics,
        "alarm_triggered": alarm_triggered,
        "samples": results,
    }

    # =============================
    # UPLOAD SNAPSHOT
    # =============================
    s3_key = f"{S3_PREFIX}/alarm_snapshot_{metrics['timestamp']}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(snapshot, indent=2),
        ContentType="application/json",
    )

    print(f"\n📦 Snapshot uploaded to s3://{BUCKET_NAME}/{s3_key}")

    if alarm_triggered:
        print("🚨 ALARM TRIGGERED: High negative sentiment detected")
    else:
        print("✅ Sentiment within acceptable range")

    # =============================
    # READ BACK FROM S3
    # =============================
    print("\n📥 Reading snapshot back from S3:\n")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    content = json.loads(obj["Body"].read())

    print(json.dumps(content["metrics"], indent=2))


# =============================
# ENTRY
# =============================
if __name__ == "__main__":
    main()

