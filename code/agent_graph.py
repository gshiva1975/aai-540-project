from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ---- SageMaker predictor ----
predictor = Predictor(endpoint_name="hf-sentiment-endpoint")
predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

def analyze_sentiment(text: str) -> dict:
    result = predictor.predict({"text": text})
    return {
        "label": result[0]["label"],
        "score": float(result[0]["score"]),
    }

# ---- Agent State ----
class AgentState(TypedDict):
    text: str
    sentiment: Optional[dict]
    response: Optional[str]

# ---- Nodes ----
def sentiment_node(state: AgentState) -> AgentState:
    sentiment = analyze_sentiment(state["text"])
    return {**state, "sentiment": sentiment}

def positive_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": "Thanks for the positive feedback! Glad things are working well."
    }

def negative_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": (
            "I’m sorry this was frustrating. The good news is the system is now live, "
            "and future runs should be much smoother. Let me know how I can help."
        )
    }

# ---- Routing Logic ----
def route_by_sentiment(state: AgentState) -> str:
    if state["sentiment"]["label"] == "NEGATIVE":
        return "negative"
    return "positive"

# ---- Graph ----
graph = StateGraph(AgentState)

graph.add_node("sentiment", sentiment_node)
graph.add_node("positive", positive_node)
graph.add_node("negative", negative_node)

graph.set_entry_point("sentiment")

graph.add_conditional_edges(
    "sentiment",
    route_by_sentiment,
    {
        "positive": "positive",
        "negative": "negative",
    }
)

graph.add_edge("positive", END)
graph.add_edge("negative", END)

agent = graph.compile()

# ---- Run ----
if __name__ == "__main__":
    result = agent.invoke({
        "text": "This deployment process was painful and slow"
    })

    print("Sentiment:", result["sentiment"])
    print("Response:", result["response"])

