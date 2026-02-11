from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = Predictor(
    endpoint_name="hf-sentiment-endpoint"
)

# REQUIRED for custom containers
predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

response = predictor.predict({
    "text": "SageMaker endpoint is finally live"
})

print(response)

