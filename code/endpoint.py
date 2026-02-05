import sagemaker
from sagemaker.model import Model

# ---- Session & role ----
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

print("Role:", role)

# ---- ECR image URI ----
image_uri = "288146132384.dkr.ecr.us-east-1.amazonaws.com/hf-sentiment:latest"

# ---- Create SageMaker Model ----
model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=sess,
)

# ---- Deploy endpoint ----
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="hf-sentiment-endpoint"
)

