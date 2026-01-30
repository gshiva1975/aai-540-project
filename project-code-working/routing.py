from transformers import pipeline

class ZeroShotRouter:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def route(self, prompt, tools: dict):
        labels = list(tools.values())
        result = self.classifier(prompt, labels)
        top_label = result["labels"][0]
        score = result["scores"][0]
        for tool, label in tools.items():
            if label == top_label:
                return tool, label, score
        return None, None, 0.0

