import boto3
import json
import time

REGION = "us-east-1"
MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"

MODEL_NAME = "sentiment-baseline-model"
ENDPOINT_NAME = "sentiment-baseline-endpoint"

ENDPOINT_CONFIG = "sentiment-baseline-config-v2"

INSTANCE_TYPE = "ml.m5.large"

# 👇 MUST be allowed via iam:PassRole
EXECUTION_ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

sm = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# -------------------------------------------------
# GET LATEST APPROVED MODEL PACKAGE
# -------------------------------------------------
pkgs = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

model_package_arn = pkgs["ModelPackageSummaryList"][0]["ModelPackageArn"]

# -------------------------------------------------
# CREATE MODEL
# -------------------------------------------------
try:
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={"ModelPackageName": model_package_arn},
        ExecutionRoleArn=EXECUTION_ROLE_ARN
    )
except sm.exceptions.ResourceInUse:
    pass

# -------------------------------------------------
# CREATE ENDPOINT CONFIG
# -------------------------------------------------
try:
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1
            }
        ]
    )
except sm.exceptions.ResourceInUse:
    pass

# -------------------------------------------------
# CREATE / UPDATE ENDPOINT
# -------------------------------------------------
try:
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG
    )
except sm.exceptions.ResourceInUse:
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG
    )

# -------------------------------------------------
# WAIT
# -------------------------------------------------
while True:
    status = sm.describe_endpoint(
        EndpointName=ENDPOINT_NAME
    )["EndpointStatus"]
    if status == "InService":
        break
    time.sleep(30)

# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
payload = {"inputs": "I absolutely loved this movie!"}

resp = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

print(json.loads(resp["Body"].read()))

