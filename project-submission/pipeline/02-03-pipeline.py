import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

# =====================================================
# CONFIG
# =====================================================
REGION = "us-east-1"
PIPELINE_NAME = "SentimentAnalysisPipeline"
MODEL_PACKAGE_GROUP = "SentimentAnalysisModels"

IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:"
    "1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04-v1.6"
)

MODEL_ARTIFACT = (
    "s3://my-sentiment-model-bucket-123456/"
    "sentiment-models/pretrained/model.tar.gz"
)

ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

# =====================================================
# SESSION (CRITICAL FIX)
# =====================================================
boto_sess = boto3.Session(region_name=REGION)
pipeline_session = PipelineSession(boto_session=boto_sess)
sm_client = boto_sess.client("sagemaker")

# =====================================================
# PIPELINE PARAMETERS
# =====================================================
model_approval = ParameterString(
    name="ModelApprovalStatus",
    default_value="Approved"
)

# =====================================================
# OPTIONAL METRICS STEP (DUMMY PLACEHOLDER)
# =====================================================
processor = ScriptProcessor(
    image_uri=IMAGE_URI,
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=ROLE_ARN,
    sagemaker_session=pipeline_session,
)

metrics_step = ProcessingStep(
    name="DummyMetricsStep",
    processor=processor,
    code="metrics.py",   # optional, can be empty
)

# =====================================================
# MODEL REGISTRATION STEP
# =====================================================
model = Model(
    image_uri=IMAGE_URI,
    model_data=MODEL_ARTIFACT,
    role=ROLE_ARN,
    sagemaker_session=pipeline_session,
)

register_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status=model_approval,
    ),
)

# =====================================================
# PIPELINE
# =====================================================
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[model_approval],
    steps=[metrics_step, register_step],
    sagemaker_session=pipeline_session,
)

# =====================================================
# EXECUTE
# =====================================================
if __name__ == "__main__":
    print(" Creating / Updating pipeline...")
    pipeline.upsert(role_arn=ROLE_ARN)

    print(" Starting pipeline execution...")
    execution = pipeline.start()

    print(f" Pipeline execution started:\n{execution.arn}")

