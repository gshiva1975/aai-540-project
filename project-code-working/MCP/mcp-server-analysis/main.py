import pandas as pd
from textblob import TextBlob
from mcp.server.fastmcp import FastMCP

# Initialize MCP app
mcp = FastMCP("iphone_sentiment")

# Load iPhone dataset
df = pd.read_csv("iphone.csv")  # Ensure the CSV is in the same folder or give full path
df = df.dropna(subset=["reviewDescription"])
df["review"] = df["reviewDescription"].astype(str)

# Sentiment logic
def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Preprocess once
df["sentiment"] = df["review"].apply(analyze_sentiment)

@mcp.tool()
async def get_docs(query: str):
    """
    Search iPhone reviews and analyze sentiment for a keyword.
    """
    filtered = df[df["review"].str.contains(query, case=False, na=False)]

    if filtered.empty:
        return f"No reviews found for '{query}'."

    sentiment_counts = filtered["sentiment"].value_counts().to_dict()
    samples = filtered.head(3)[["review", "sentiment"]].to_dict(orient="records")

    result = f" Sentiment Summary for '{query}':\n"
    for sentiment, count in sentiment_counts.items():
        result += f"- {sentiment}: {count}\n"

    result += "\n Sample Reviews:\n"
    for ex in samples:
        result += f"- {ex['sentiment']}: {ex['review'][:150]}...\n"

    return result.strip()

if __name__ == "__main__":
    mcp.run(transport="stdio")

