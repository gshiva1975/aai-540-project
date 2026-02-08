import boto3, json

ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="sentiment-baseline-endpoint",
    ContentType="application/json",
    Body=json.dumps({"inputs": "I love this product!"})
)

print(response["Body"].read().decode())

