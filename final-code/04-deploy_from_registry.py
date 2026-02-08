import time
import json
import boto3
from botocore.exceptions import ClientError

# =====================================================
# CONFIG
# =====================================================
REGION = "us-east-1"

MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"

MODEL_NAME = f"sentiment-baseline-model-v{int(time.time())}"
ENDPOINT_CONFIG_NAME = f"sentiment-baseline-config-v{int(time.time())}"
ENDPOINT_NAME = "sentiment-baseline-endpoint"

# IMPORTANT:
# Use the SAME role you are running as (LabRole)
EXECUTION_ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

INSTANCE_TYPE = "ml.m5.large"

# =====================================================
# CLIENTS
# =====================================================
sm = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# =====================================================
# 1️⃣ GET LATEST APPROVED MODEL PACKAGE
# =====================================================
print("🔍 Fetching latest APPROVED model package...")

packages = sm.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)

if not packages["ModelPackageSummaryList"]:
    raise RuntimeError("❌ No approved model package found")

MODEL_PACKAGE_ARN = packages["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"✅ Using Model Package: {MODEL_PACKAGE_ARN}")

# =====================================================
# 2️⃣ CREATE MODEL (FROM REGISTRY ONLY)
# =====================================================
print("📦 Creating SageMaker Model...")

try:
    sm.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
        Containers=[
            {
                "ModelPackageName": MODEL_PACKAGE_ARN
            }
        ],
    )
    print("✅ Model created")

except ClientError as e:
    if "already exists" in str(e):
        print("⚠️ Model already exists, continuing")
    else:
        raise e

# =====================================================
# 3️⃣ CREATE ENDPOINT CONFIG
# =====================================================
print("⚙️ Creating endpoint config...")

try:
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InitialInstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    print("✅ Endpoint config created")

except ClientError as e:
    if "already exists" in str(e):
        print("⚠️ Endpoint config already exists, continuing")
    else:
        raise e

# =====================================================
# 4️⃣ CREATE OR UPDATE ENDPOINT (CORRECTLY)
# =====================================================
print("🚀 Deploying endpoint...")

endpoint_exists = False

try:
    sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    endpoint_exists = True
    print("🔄 Endpoint exists → updating")

except ClientError as e:
    if "Could not find endpoint" in str(e):
        print("🆕 Endpoint does not exist → creating")
    else:
        raise e

if endpoint_exists:
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
else:
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )

# =====================================================
# 5️⃣ WAIT FOR ENDPOINT
# =====================================================
print("⏳ Waiting for endpoint to be InService...")

waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=ENDPOINT_NAME)

print("✅ Endpoint is InService")

# =====================================================
# 6️⃣ TEST INFERENCE
# =====================================================
print("🧪 Testing inference...")

payload = {
    "inputs": "I absolutely loved this product!"
}

response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = response["Body"].read().decode("utf-8")

print("🎯 Inference result:")
print(result)

print("\n🏁 Deployment completed successfully")

