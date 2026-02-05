from langchain_core.tools import Tool
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# ---- SageMaker predictor ----
predictor = Predictor(endpoint_name="hf-sentiment-endpoint")
predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

def sentiment_tool(text: str):
    return predictor.predict({"text": text})

tool = Tool(
    name="SentimentAnalysis",
    description="Analyze sentiment using SageMaker endpoint",
    func=sentiment_tool,
)

if __name__ == "__main__":
    result = tool.invoke("LangChain v0.2 tools finally work")
    print(result)

