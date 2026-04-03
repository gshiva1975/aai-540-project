import boto3

REGION = "us-east-1"
ENDPOINT_PREFIX = "sentiment"
MODEL_PREFIX = "sentiment"
MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"
BUCKET = "my-sentiment-model-bucket-123456"

ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"


sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

print(" Cleaning SageMaker resources...")

# Delete endpoints
for ep in sm.list_endpoints()["Endpoints"]:
    if ep["EndpointName"].startswith(ENDPOINT_PREFIX):
        sm.delete_endpoint(EndpointName=ep["EndpointName"])
        print("Deleted endpoint:", ep["EndpointName"])

# Delete endpoint configs
for cfg in sm.list_endpoint_configs()["EndpointConfigs"]:
    if cfg["EndpointConfigName"].startswith(MODEL_PREFIX):
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print("Deleted endpoint config:", cfg["EndpointConfigName"])

# Delete models
for model in sm.list_models()["Models"]:
    if model["ModelName"].startswith(MODEL_PREFIX):
        sm.delete_model(ModelName=model["ModelName"])
        print("Deleted model:", model["ModelName"])

print(" Cleanup complete")

