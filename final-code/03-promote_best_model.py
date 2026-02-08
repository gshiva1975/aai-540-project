import boto3, json

REGION = "us-east-1"
GROUP = "SentimentAnalysisModels"
ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

best = None
best_acc = -1

packages = sm.list_model_packages(ModelPackageGroupName=GROUP)["ModelPackageSummaryList"]

for p in packages:
    desc = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
    metrics = desc.get("ModelMetrics", {})
    if not metrics:
        continue

    uri = metrics["ModelQuality"]["Statistics"]["S3Uri"]
    bucket, key = uri.replace("s3://", "").split("/", 1)
    data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

    if data["accuracy"] > best_acc:
        best_acc = data["accuracy"]
        best = p["ModelPackageArn"]

if not best:
    raise RuntimeError("No valid model metrics found")

sm.update_model_package(ModelPackageArn=best, ModelApprovalStatus="Approved")
print("🏆 Promoted:", best)

