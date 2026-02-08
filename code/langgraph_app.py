import json
import boto3
from langgraph.graph import StateGraph
from typing import TypedDict

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

class State(TypedDict):
    text: str
    result: str

def sentiment_node(state: State):
    response = runtime.invoke_endpoint(
        EndpointName="sentiment-baseline-endpoint",
        ContentType="application/json",
        Body=json.dumps({"inputs": state["text"]})
    )
    return {
        "result": response["Body"].read().decode()
    }

graph = StateGraph(State)
graph.add_node("sentiment", sentiment_node)
graph.set_entry_point("sentiment")
graph.set_finish_point("sentiment")

app = graph.compile()

print(app.invoke({"text": "This product is amazing!"}))

