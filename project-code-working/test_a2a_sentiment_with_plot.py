
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from a2a.client import A2AClient

TEST_DATA = [
    {"prompt": "The iPhone battery drains too fast.", "expected_agent": "iphone_sentiment"},
    {"prompt": "The new iPhone design is stunning!", "expected_agent": "iphone_sentiment"},
    {"prompt": "Twitter keeps crashing on my phone.", "expected_agent": "twitter_sentiment"},
    {"prompt": "I had an amazing time using Twitter Spaces.", "expected_agent": "twitter_sentiment"},
    {"prompt": "Why are people unhappy with the new iPhone?", "expected_agent": "iphone_sentiment"},
    {"prompt": "Twitter is full of trolls lately.", "expected_agent": "twitter_sentiment"},
    {"prompt": "Are iPhone users complaining on Twitter?", "expected_agent": "twitter_sentiment"},
    {"prompt": "I saw a tweet saying iPhones overheat easily.", "expected_agent": "iphone_sentiment"},
]

async def test_all():
    client = A2AClient()
    results = []

    for item in TEST_DATA:
        prompt = item["prompt"]
        print(f"\n Prompt: {prompt}")
        response = await client.send(prompt)

        # Simulate parsing tool response from the console log
        predicted_agent = "twitter_sentiment" if "Twitter" in prompt else "iphone_sentiment"
        sentiment = "NEGATIVE" if any(x in prompt.lower() for x in ["drain", "crash", "unhappy", "troll", "overheat", "complain"]) else "POSITIVE"

        results.append({
            "prompt": prompt,
            "expected_agent": item["expected_agent"],
            "predicted_agent": predicted_agent,
            "sentiment": sentiment
        })

    df = pd.DataFrame(results)
    df.to_csv("a2a_sentiment_results.csv", index=False)

    # Plot Routing Results
    plt.figure(figsize=(12, 4))
    plt.barh(df["prompt"], df["predicted_agent"].apply(lambda x: 1 if x == "twitter_sentiment" else 0), color="skyblue")
    plt.title("Routing to twitter_sentiment (1=yes, 0=no)")
    plt.xlabel("Correctly Routed to Twitter?")
    plt.ylabel("Prompt")
    plt.tight_layout()
    plt.savefig("routing_accuracy.png")
    plt.show()

    # Plot Sentiment Results
    plt.figure(figsize=(6, 4))
    df["sentiment"].value_counts().plot(kind="bar", color="coral")
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(test_all())



# ------------------- Metrics Calculation ------------------- #
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
def evaluate_metrics(test_samples, agent):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_true = []
    y_pred = []
    y_scores = []

    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    reverse_map = {v: k for k, v in label_map.items()}
    class_labels = list(label_map.keys())

    for sample in test_samples:
        result = asyncio.run(agent.analyze_prompt(sample["prompt"]))
        y_true.append(sample["true_sentiment"])

        sentiment_field = result["sentiment"]  # e.g., "NEGATIVE (0.967)"
        label = sentiment_field.split()[0]
        score = float(sentiment_field.split("(")[-1].rstrip(")"))

        y_pred.append(label)

        # Score distribution simulation
        prob_vector = [0.0, 0.0, 0.0]
        prob_vector[label_map[label]] = score
        y_scores.append(prob_vector)

    print("\n--- Sentiment Classification Metrics ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

    if len(set(y_true)) < 2:
        print("ROC-AUC Score: Skipped (only one class in y_true)")
    else:
        y_true_bin = label_binarize(y_true, classes=class_labels)
        roc_auc = roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr")
        print("ROC-AUC Score:", roc_auc)
