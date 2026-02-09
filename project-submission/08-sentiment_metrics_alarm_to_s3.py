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

SAMPLE_SIZE = 200           # random samples
MAX_CHARS = 1000           # prevents 512-token crash
NEG_THRESHOLD = 0.60       # alarm threshold

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
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )

def prepare_text(title, description):
    text = f"{title}. {description}"
    return text[:MAX_CHARS]

def map_label(label):
    return "positive" if label == "LABEL_1" else "negative"

def invoke_endpoint(text):
    payload = {"inputs": text}

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read())
    label = result[0]["label"]
    score = float(result[0]["score"])

    return {
        "label": label,
        "sentiment": map_label(label),
        "confidence": round(score, 4),
    }

# =============================
# LOAD & SAMPLE DATA
# =============================
def load_reviews(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"reviewTitle", "reviewDescription"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}"
        )

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

    for i, r in enumerate(reviews, start=1):
        text = prepare_text(r["reviewTitle"], r["reviewDescription"])
        prediction = invoke_endpoint(text)

        sentiments.append(prediction["sentiment"])

        record = {
            "reviewTitle": r["reviewTitle"],
            "reviewDescription": r["reviewDescription"],
            "prediction": prediction,
        }
        results.append(record)

        print(f"{i}. [{prediction['sentiment'].upper()} | {prediction['confidence']}]")
        print(f"   {r['reviewTitle']}")
    tests = [ "Amazing", "Excellent product", "Best phone of the decade", "Waste of money" ]
    for t in tests:
        prediction = invoke_endpoint(t)
        sentiments.append(prediction["sentiment"])

        record = {
            "reviewTitle": t,
            "prediction": prediction,
        }
        results.append(record)

        print(f"{i}. [{prediction['sentiment'].upper()} | {prediction['confidence']}]")

    # =============================
    # METRICS
    # =============================
    counts = Counter(sentiments)
    total = sum(counts.values())

    metrics = {
        "total_reviews": total,
        "positive_count": counts.get("positive", 0),
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

