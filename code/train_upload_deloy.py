import os
import tarfile
import boto3
from botocore.exceptions import ClientError
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =================================================
# CONFIG
# =================================================
REGION = "us-east-1"
S3_BUCKET = "my-sentiment-model-bucket-123456"   # must be unique
S3_PREFIX = "sentiment-models/pretrained"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"

LOCAL_MODEL_DIR = "./model"
MODEL_TAR_PATH = "./model.tar.gz"

# AWS-managed Hugging Face inference image (CRITICAL)
INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.36.0-cpu-py310"
)

INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39"
)

INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:"
    "1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"
)

# =================================================
# AWS CLIENTS
# =================================================
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

METRICS_S3_URI = (
    "s3://my-sentiment-model-bucket-123456/metrics/metrics_v3.json"
)
# =================================================
# CREATE S3 BUCKET (IDEMPOTENT)
# =================================================
def ensure_bucket(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            if REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": REGION}
                )

ensure_bucket(S3_BUCKET)

# =================================================
# LOAD PRETRAINED MODEL (NO TRAINING)
# =================================================
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.save_pretrained(LOCAL_MODEL_DIR)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)

# =================================================
# PACKAGE MODEL
# =================================================
with tarfile.open(MODEL_TAR_PATH, "w:gz") as tar:
    tar.add(LOCAL_MODEL_DIR, arcname=".")

# =================================================
# UPLOAD TO S3
# =================================================
s3_key = f"{S3_PREFIX}/model.tar.gz"
s3.upload_file(MODEL_TAR_PATH, S3_BUCKET, s3_key)

MODEL_DATA_URL = f"s3://{S3_BUCKET}/{s3_key}"

# =================================================
# ENSURE MODEL PACKAGE GROUP
# =================================================
try:
    sm.describe_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP
    )
except sm.exceptions.ResourceNotFound:
    sm.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageGroupDescription="Sentiment analysis models"
    )

# =================================================
# REGISTER BASELINE MODEL
# =================================================

response = sm.create_model_package(
    ModelPackageGroupName="SentimentAnalysisModels",
    ModelPackageDescription="Baseline model with metrics",
    InferenceSpecification={
        "Containers": [
            {
                "Image": INFERENCE_IMAGE_URI,
                "ModelDataUrl": MODEL_DATA_URL
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"]
    },
    ModelApprovalStatus="PendingManualApproval",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": METRICS_S3_URI
            }
        }
    }
)

print("📌 Model Package ARN:", response["ModelPackageArn"])


print("✅ BASELINE MODEL REGISTERED")
print(response["ModelPackageArn"])

