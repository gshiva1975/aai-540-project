from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Create predictor once (reuse connection)
predictor = Predictor(endpoint_name="hf-sentiment-endpoint")
predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

def sentiment_tool(text: str) -> dict:
    """
    Calls the SageMaker sentiment analysis endpoint.
    """
    result = predictor.predict({"text": text})
    return {
        "label": result[0]["label"],
        "score": float(result[0]["score"]),
    }

