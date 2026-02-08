Steps 

TERMINAL 1

docker build -t hf-sentiment:latest .




sh-4.2$ aws sts get-caller-identity
{
    "UserId": "AROAUGFWZBWQG4BF4QMMC:SageMaker",
    "Account": "288146132384",
    "Arn": "arn:aws:sts::288146132384:assumed-role/LabRole/SageMaker"
}
sh-4.2$ 


sh-4.2$ aws configure get region
us-east-1

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=288146132384
export REPO_NAME=aai-540-hf-training-1

sh-4.2$  aws ecr create-repository   --repository-name $REPO_NAME   --region $AWS_REGION
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:288146132384:repository/aai-540-hf-training-1",
        "registryId": "288146132384",
        "repositoryName": "aai-540-hf-training-1",
        "repositoryUri": "288146132384.dkr.ecr.us-east-1.amazonaws.com/aai-540-hf-training-1",
        "createdAt": 1770436445.865,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }



sh-4.2$ aws ecr get-login-password   --region ${AWS_REGION}   | docker login   --username AWS    --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
WARNING! Your password will be stored unencrypted in /home/ec2-user/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded


docker tag hf-sentiment:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hf-sentiment:latest


docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/hf-sentiment:latest

python endpoint.py

sh-4.2$ python endpoint.py 
Unable to load JumpStart region config.
Traceback (most recent call last):
  File "/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/constants.py", line 69, in _load_region_config
    with open(filepath) as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/region_config.json'
sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
Role: arn:aws:iam::288146132384:role/LabRole




pip3 install transformers

python serve.py serve

sh-4.2$ python serve.py serve
config.json: 100%|████████████████████████████████████████████████████████████████| 629/629 [00:00<00:00, 1.45MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
tokenizer_config.json: 100%|█████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 187kB/s]
vocab.txt: 100%|████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 25.6MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████| 268M/268M [00:01<00:00, 222MB/s]
Loading weights: 100%|██████████████| 104/104 [00:00<00:00, 1592.50it/s, Materializing param=pre_classifier.weight]
INFO:     Started server process [5361]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)


TERMINAL 2

aws sagemaker describe-endpoint \
  --endpoint-name hf-sentiment-endpoint \
  --query "EndpointStatus"


sh-4.2$ aws sagemaker describe-endpoint \
>   --endpoint-name hf-sentiment-endpoint \
>   --query "EndpointStatus"
"InService"
sh-4.2$ 


python3 -m pip install langgraph

python agent_graph.py



sh-4.2$ python agent_graph.py
Unable to load JumpStart region config.
Traceback (most recent call last):
  File "/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/constants.py", line 69, in _load_region_config
    with open(filepath) as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/region_config.json'
sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
Sentiment: {'label': 'NEGATIVE', 'score': 0.9994339346885681}
Response: I’m sorry this was frustrating. The good news is the system is now live, and future runs should be much smoother. Let me know how I can help.
sh-4.2$ 



sh-4.2$ python agent_graph.py
Unable to load JumpStart region config.
Traceback (most recent call last):
  File "/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/constants.py", line 69, in _load_region_config
    with open(filepath) as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.10/site-packages/sagemaker/jumpstart/region_config.json'
sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /home/ec2-user/.config/sagemaker/config.yaml
Sentiment: {'label': 'NEGATIVE', 'score': 0.9996519088745117}
Response: I’m sorry this was frustrating. The good news is the system is now live, and future runs should be much smoother. Let me know how I can help.



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





fine_trian.py Trains a sentiment model (SST-2, DistilBERT)
Dockerfile + app.py + serve.py Packages a FastAPI inference server and deploys it to SageMaker
endpoint.py / sess.py / tools.py / agent_graph.py Call the deployed endpoint from Python, tools, or a LangGraph agent

