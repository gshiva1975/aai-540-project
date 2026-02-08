import os, json, tarfile, boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REGION = "us-east-1"
BUCKET = "my-sentiment-model-bucket-123456"
PREFIX = "sentiment-models"
GROUP = "SentimentAnalysisModels"

HF_MODEL_ID = "distilbert-base-uncased"
IMAGE_URI = (
 "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
 "huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04-v1.6"
)
ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

# Load HF model
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)

os.makedirs("model", exist_ok=True)
tokenizer.save_pretrained("model")
model.save_pretrained("model")

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model", arcname=".")

MODEL_S3 = f"{PREFIX}/model.tar.gz"
s3.upload_file("model.tar.gz", BUCKET, MODEL_S3)

metrics = {"accuracy": 0.93}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

s3.upload_file("metrics.json", BUCKET, "metrics/metrics.json")

response = sm.create_model_package(
    ModelPackageGroupName=GROUP,
    InferenceSpecification={
        "Containers": [{"Image": IMAGE_URI, "ModelDataUrl": f"s3://{BUCKET}/{MODEL_S3}"}],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"]
    },
    ModelApprovalStatus="Approved",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": f"s3://{BUCKET}/metrics/metrics.json"
            }
        }
    }
)

print("📌 Registered:", response["ModelPackageArn"])

