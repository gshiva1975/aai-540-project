
import time, boto3, sagemaker

REGION = "us-east-1"
GROUP = "SentimentAnalysisModels"
ENDPOINT = "sentiment-baseline-endpoint"
ROLE = "arn:aws:iam::288146132384:role/AmazonSageMaker-ExecutionRole"
IMAGE_URI = (
 "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
 "huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04-v1.6"
)

sm = boto3.client("sagemaker", region_name=REGION)
sess = sagemaker.Session()

pkg = sm.list_model_packages(
    ModelPackageGroupName=GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)["ModelPackageSummaryList"][0]

desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
model_data = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

model_name = f"sentiment-baseline-model-v{int(time.time())}"
cfg = f"{model_name}-cfg"

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=ROLE,
    PrimaryContainer={"Image": IMAGE_URI, "ModelDataUrl": model_data}
)

sm.create_endpoint_config(
    EndpointConfigName=cfg,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1
    }]
)

try:
    sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=cfg)
except sm.exceptions.ResourceNotFound:
    sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=cfg)

sess.wait_for_endpoint(ENDPOINT)
print("🚀 Deployed:", ENDPOINT)

