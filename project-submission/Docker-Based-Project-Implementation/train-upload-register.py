import os
import tarfile
import boto3
from botocore.exceptions import ClientError
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =================================================
# CONFIG (EDIT IF NEEDED)
# =================================================
REGION = "us-east-1"

S3_BUCKET = "my-sentiment-model-bucket-123456"   # MUST be globally unique
S3_PREFIX = "sentiment-models/pretrained"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"

LOCAL_MODEL_DIR = "./model"
MODEL_TAR_PATH = "./model.tar.gz"

# =================================================
# AWS CLIENTS
# =================================================
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

#  AWS-managed PyTorch inference image (DO NOT CHANGE)
INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-inference:2.0.0-cpu-py310"
)

# =================================================
# CREATE S3 BUCKET (SAFE / IDEMPOTENT)
# =================================================
def ensure_bucket_exists(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f" S3 bucket exists: {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f" Creating S3 bucket: {bucket}")
            if REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={
                        "LocationConstraint": REGION
                    }
                )
        else:
            raise

ensure_bucket_exists(S3_BUCKET)

# =================================================
# LOAD PRETRAINED MODEL (NO TRAINING)
# =================================================
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.save_pretrained(LOCAL_MODEL_DIR)
tokenizer.save_pretrained(LOCAL_MODEL_DIR)

print(" Pretrained model saved locally")

# =================================================
# PACKAGE MODEL (SageMaker-compatible)
# =================================================
with tarfile.open(MODEL_TAR_PATH, "w:gz") as tar:
    tar.add(LOCAL_MODEL_DIR, arcname=".")

print(" model.tar.gz created")

# =================================================
# UPLOAD TO S3
# =================================================
s3_key = f"{S3_PREFIX}/model.tar.gz"
s3.upload_file(MODEL_TAR_PATH, S3_BUCKET, s3_key)

MODEL_DATA_URL = f"s3://{S3_BUCKET}/{s3_key}"
print(f" Model uploaded to {MODEL_DATA_URL}")

# =================================================
# ENSURE MODEL PACKAGE GROUP EXISTS
# =================================================
def ensure_model_package_group():
    try:
        sm.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP
        )
        print(" Model Package Group exists")
    except sm.exceptions.ResourceNotFound:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Sentiment analysis models"
        )
        print(" Model Package Group created")

ensure_model_package_group()

# =================================================
# REGISTER BASELINE MODEL
# =================================================
response = sm.create_model_package(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelPackageDescription="BASELINE MODEL – initial reference version",
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
    ModelApprovalStatus="Approved",
    CustomerMetadataProperties={
        "role": "baseline",
        "stage": "production-reference"
    }
)

print(" BASELINE MODEL REGISTERED")
print(f" Model Package ARN: {response['ModelPackageArn']}")

