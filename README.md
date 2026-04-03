








<img width="609" height="511" alt="Screenshot 2026-04-03 at 10 08 56 AM" src="https://github.com/user-attachments/assets/a6ca9d28-e3f7-4b68-af8d-e5137fbf3978" />









































# aai-540-project

path - /Users/gshiva/aa-54--jan30/AAI-510/aai-540-project/project-code-working


# Using the Project TAR in Amazon SageMaker

This section explains how to deploy and run the iPhone & Twitter Aspect-Based Sentiment Analysis (ABSA) system using the provided project archive in an Amazon SageMaker environment.

---

# Using the Project TAR in Amazon SageMaker

This section explains how to deploy and run the iPhone & Twitter Aspect-Based Sentiment Analysis (ABSA) system using the provided project archive in an Amazon SageMaker environment.

---

## 0. Create an S3 Bucket (Required)

Before running any pipelines, create an S3 bucket to store datasets, training outputs, and model artifacts.

```python
import boto3
from botocore.exceptions import ClientError

region = "us-east-1"
bucket_name = "absa-iphone-twitter-<your-unique-suffix>"

s3 = boto3.client("s3", region_name=region)

try:
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region}
        )
    print(f"S3 bucket created: {bucket_name}")
except ClientError as e:
    print(f"Bucket already exists or error occurred: {e}")


---

## 1. Upload the Project Archive

### Option A: SageMaker Studio (Recommended)
1. Open Amazon SageMaker Studio.
2. In the File Browser, upload:
   iphone-twitter-absa-sagemaker-full.tar
3. Open a Terminal in SageMaker Studio and extract the project:
   ```bash
   tar -xvf iphone-twitter-absa-sagemaker-full.tar
   cd iphone-twitter-absa-sagemaker

   
Option B: Upload via S3
bash

aws s3 cp iphone-twitter-absa-sagemaker-full.tar s3://<your-bucket>/absa/code/
aws s3 cp s3://<your-bucket>/absa/code/iphone-twitter-absa-sagemaker-full.tar .


tar -xvf iphone-twitter-absa-sagemaker-full.tar


2. Install Dependencies
Use a PyTorch / Hugging Face kernel in SageMaker Studio, then run:

bash

pip install -r requirements.txt
3. Upload Datasets to S3
iPhone Reviews Dataset

bash

aws s3 cp iphone.csv s3://<your-bucket>/absa/iphone/raw/

Twitter Dataset
bash

aws s3 cp twitter.csv s3://<your-bucket>/absa/twitter/raw/

4. Run Data Preprocessing
Preprocess raw data into training and validation sets:

bash

python preprocessing/preprocess_iphone.py
python preprocessing/preprocess_twitter.py
Upload processed files:

bash

aws s3 cp train.csv s3://<your-bucket>/absa/iphone/train/
aws s3 cp validation.csv s3://<your-bucket>/absa/iphone/validation/
5. Execute SageMaker Pipeline (Train → Register)
Edit the S3 paths in pipelines/pipeline.py:

python

inputs={"train": "s3://<your-bucket>/absa/iphone/train"}
Run the pipeline:

bash

python pipelines/pipeline.py
This pipeline trains the model, stores artifacts in S3, and registers the model in the SageMaker Model Registry.

6. Approve the Model in Model Registry
Open the SageMaker Console.

Navigate to Model Registry.

Select the appropriate Model Package Group.

Approve the latest model version.

7. Deploy the Model Endpoint
python

from sagemaker.huggingface import HuggingFaceModel
import sagemaker

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

model = HuggingFaceModel(
    model_data="<S3_MODEL_ARTIFACT_URI>",
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39"
)

predictor = model.deploy(
    endpoint_name="iphone-sentiment-endpoint",
    instance_type="ml.m5.xlarge",
    initial_instance_count=1
)
Repeat this step for the Twitter model using a separate endpoint name.

8. Run LangGraph Agent for Inference
bash

python agent/langgraph_agent.py
The agent routes requests to the correct SageMaker endpoint, returns sentiment and confidence, logs results to Feature Store, and supports SHAP explainability.

9. Feature Store Logging (Optional)
python

from feature_store.log_inference import log_prediction

log_prediction(
    text="Battery drains fast",
    sentiment="NEGATIVE",
    confidence=0.94
)
10. SHAP Explainability
bash

python shap/explain.py
This generates token-level explanations for transparency and auditability.

11. Required IAM Permissions
Ensure the SageMaker execution role includes:

AmazonSageMakerFullAccess

AmazonS3FullAccess

AmazonSageMakerFeatureStoreAccess

Summary
This setup enables a fully governed ML lifecycle using Amazon SageMaker, including data ingestion, training, model registration, deployment, agentic inference with LangGraph, Feature Store logging, and SHAP-based explainability.









