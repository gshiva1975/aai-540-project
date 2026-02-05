
Order of Execution (End-to-End)

Model Fine-Tuning (One-Time / Occasional)
Fine-tuning is performed offline to update model weights. This step trains the base model on your dataset and saves the fine-tuned artifacts locally.

python finetune.py


Output:

model/
├── pytorch_model.bin
├── config.json
└── tokenizer files


Docker Image Build (Per Model Version)
A Docker image is built that packages the inference code and the fine-tuned model artifacts into an immutable runtime.

docker build -t hf-sentiment:latest .


Push Image to Amazon ECR
The Docker image is pushed to ECR so that SageMaker can securely pull it.

aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker tag hf-sentiment:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hf-sentiment:latest

docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hf-sentiment:latest


SageMaker Endpoint Deployment
SageMaker provisions managed compute, pulls the image from ECR, and starts the container.

python deploy_endpoint.py


(Internally executes model.deploy().)

Container Startup (Once per Instance)
On each instance, SageMaker runs the container with the serve command:

python serve.py serve


This launches the inference server and loads the model into memory once per instance.

Health Checks and Endpoint Activation
SageMaker calls the /ping endpoint until the container is healthy and the endpoint enters the InService state.

aws sagemaker describe-endpoint \
  --endpoint-name hf-sentiment-endpoint \
  --query "EndpointStatus"


Inference Execution (Per Request)
Clients send requests to the live endpoint. Each request triggers tokenization and a forward pass of the model.

from sagemaker.predictor import Predictor

predictor = Predictor("hf-sentiment-endpoint")
predictor.predict({"text": "This system works well"})


Agent Routing and Response Handling (Per Request)
The LangGraph agent consumes the inference result, applies conditional routing logic, and returns the final response.

python agent_graph.py


When the endpoint is no longer needed, clean up SageMaker and ECR resources to prevent ongoing charges.

1. Delete the SageMaker Endpoint

This stops the managed EC2 instances backing the endpoint.

aws sagemaker delete-endpoint \
  --endpoint-name hf-sentiment-endpoint \
  --region us-east-1


(Optional) Verify deletion:

aws sagemaker describe-endpoint \
  --endpoint-name hf-sentiment-endpoint

2. Delete the Endpoint Configuration
aws sagemaker delete-endpoint-config \
  --endpoint-config-name hf-sentiment-endpoint \
  --region us-east-1

3. Delete the SageMaker Model
aws sagemaker delete-model \
  --model-name hf-sentiment-endpoint \
  --region us-east-1

4. Delete the ECR Image (Optional but Recommended)

If the image is no longer required:

aws ecr batch-delete-image \
  --repository-name hf-sentiment \
  --image-ids imageTag=latest \
  --region us-east-1


(Optional) Delete the entire repository:

aws ecr delete-repository \
  --repository-name hf-sentiment \
  --force \
  --region us-east-1


Note: Deleting the repository removes all image versions.

Makefile (End-to-End Automation)

Create a file named Makefile in the project root:

# -------- Configuration --------
ACCOUNT_ID ?= 288146132384
REGION ?= us-east-1
REPO_NAME ?= hf-sentiment
IMAGE_TAG ?= latest
IMAGE_URI = $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPO_NAME):$(IMAGE_TAG)
ENDPOINT_NAME ?= hf-sentiment-endpoint

# -------- Targets --------

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  finetune        Run model fine-tuning"
	@echo "  build           Build Docker image"
	@echo "  login           Login to Amazon ECR"
	@echo "  push            Tag and push image to ECR"
	@echo "  deploy          Deploy SageMaker endpoint"
	@echo "  status          Check endpoint status"
	@echo "  predict         Run a sample prediction"
	@echo "  cleanup         Delete endpoint, config, model, and ECR image"

.PHONY: finetune
finetune:
	python finetune.py

.PHONY: build
build:
	docker build -t $(REPO_NAME):$(IMAGE_TAG) .

.PHONY: login
login:
	aws ecr get-login-password --region $(REGION) \
	| docker login --username AWS --password-stdin \
	  $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com

.PHONY: push
push: login
	docker tag $(REPO_NAME):$(IMAGE_TAG) $(IMAGE_URI)
	docker push $(IMAGE_URI)

.PHONY: deploy
deploy:
	python deploy_endpoint.py

.PHONY: status
status:
	aws sagemaker describe-endpoint \
	  --endpoint-name $(ENDPOINT_NAME) \
	  --region $(REGION) \
	  --query "EndpointStatus"

.PHONY: predict
predict:
	python agent_graph.py

.PHONY: cleanup
cleanup:
	-aws sagemaker delete-endpoint --endpoint-name $(ENDPOINT_NAME) --region $(REGION)
	-aws sagemaker delete-endpoint-config --endpoint-config-name $(ENDPOINT_NAME) --region $(REGION)
	-aws sagemaker delete-model --model-name $(ENDPOINT_NAME) --region $(REGION)
	-aws ecr batch-delete-image --repository-name $(REPO_NAME) \
	  --image-ids imageTag=$(IMAGE_TAG) --region $(REGION)

Typical Usage Flow
make finetune
make build
make push
make deploy
make status
make predict
make cleanup
