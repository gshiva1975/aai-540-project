import json
import time
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

# =====================================================
# CONFIG
# =====================================================
REGION = "us-east-1"
NAMESPACE = "SentimentInference"
ENDPOINT_NAME = "sentiment-baseline-endpoint"

S3_BUCKET = "my-sentiment-monitoring-bucket"
S3_PREFIX = "alarms"

NEGATIVE_THRESHOLD = 70.0
LOW_CONF_THRESHOLD = 30.0
LOW_CONF_SCORE = 0.60

# =====================================================
# CLIENTS
# =====================================================
cloudwatch = boto3.client("cloudwatch", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

# =====================================================
# ENSURE S3 BUCKET EXISTS
# =====================================================
def ensure_bucket_exists(bucket_name: str):
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ S3 bucket exists: {bucket_name}")

    except ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchBucket"]:
            print(f"🪣 Creating S3 bucket: {bucket_name}")

            if REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": REGION
                    },
                )
            print(f"✅ S3 bucket created: {bucket_name}")
        else:
            raise e

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
# PUBLISH METRICS
# =====================================================
def publish_metrics(positive, negative, low_conf):
    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {"MetricName": "PositiveRate", "Value": positive, "Unit": "Percent"},
            {"MetricName": "NegativeRate", "Value": negative, "Unit": "Percent"},
            {"MetricName": "LowConfidenceRate", "Value": low_conf, "Unit": "Percent"},
        ],
    )

# =====================================================
# RUN INFERENCE + METRIC COLLECTION
# =====================================================
def run_inference(texts):
    pos = neg = low_conf = 0

    for text in texts:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"inputs": text}),
        )

        result = json.loads(response["Body"].read().decode())[0]
        label = normalize_label(result["label"])
        score = result["score"]

        if score < LOW_CONF_SCORE:
            low_conf += 1

        if label == "NEGATIVE":
            neg += 1
        else:
            pos += 1

    total = len(texts)

    metrics = {
        "PositiveRate": (pos / total) * 100,
        "NegativeRate": (neg / total) * 100,
        "LowConfidenceRate": (low_conf / total) * 100,
    }

    publish_metrics(
        metrics["PositiveRate"],
        metrics["NegativeRate"],
        metrics["LowConfidenceRate"],
    )

    return metrics

# =====================================================
# CREATE CLOUDWATCH ALARMS
# =====================================================
def create_alarm(metric_name, threshold):
    alarm_name = f"{metric_name}-Alarm"

    cloudwatch.put_metric_alarm(
        AlarmName=alarm_name,
        Namespace=NAMESPACE,
        MetricName=metric_name,
        Statistic="Average",
        Period=300,
        EvaluationPeriods=1,
        Threshold=threshold,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
    )

    return {
        "AlarmName": alarm_name,
        "Metric": metric_name,
        "Threshold": threshold,
    }

# =====================================================
# WRITE SNAPSHOT TO S3
# =====================================================
def write_snapshot_to_s3(metrics, alarms):
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": ENDPOINT_NAME,
        "metrics": metrics,
        "alarms": alarms,
    }

    key = f"{S3_PREFIX}/alarm_snapshot_{int(time.time())}.json"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )

    print(f"📁 Alarm snapshot uploaded to s3://{S3_BUCKET}/{key}")
    return key

# =====================================================
# READ SNAPSHOT BACK FROM S3  ✅ NEW
# =====================================================
def read_snapshot_from_s3(key: str):
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("🚀 Running sentiment inference + metrics collection")

    ensure_bucket_exists(S3_BUCKET)

    sample_texts = [
        "This phone is amazing!",
        "Worst phone I have ever used",
        "Battery life is okay, nothing special",
        "Camera quality is excellent",
    ]

    metrics = run_inference(sample_texts)

    alarms = [
        create_alarm("NegativeRate", NEGATIVE_THRESHOLD),
        create_alarm("LowConfidenceRate", LOW_CONF_THRESHOLD),
    ]

    snapshot_key = write_snapshot_to_s3(metrics, alarms)

    # 🔹 Display JSON from S3
    print("\n📄 Alarm snapshot read back from S3:\n")
    snapshot = read_snapshot_from_s3(snapshot_key)
    print(json.dumps(snapshot, indent=2))

    print("\n✅ Metrics, alarms, S3 snapshot, and display complete")

