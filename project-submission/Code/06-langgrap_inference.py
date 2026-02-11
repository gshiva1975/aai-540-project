import json, boto3
from typing import TypedDict
from langgraph.graph import StateGraph

ROLE_ARN = "arn:aws:iam::288146132384:role/LabRole"

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

def sentiment_tool(text: str):
    resp = runtime.invoke_endpoint(
        EndpointName="sentiment-baseline-endpoint",
        ContentType="application/json",
        Body=json.dumps({"inputs": text})
    )
    return json.loads(resp["Body"].read())

class State(TypedDict):
    text: str
    sentiment: dict

def node(state: State):
    return {"text": state["text"], "sentiment": sentiment_tool(state["text"])}

g = StateGraph(State)
g.add_node("sentiment", node)
g.set_entry_point("sentiment")
g.set_finish_point("sentiment")

app = g.compile()
print(app.invoke({"text": "This product is amazing!"}))

