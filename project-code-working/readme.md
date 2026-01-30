# Code Details
# a2a_main.py - Routing Agent ( Orchestrator )
# test_a2a_sentiment_with_plot.py  - Query from the User
# a2a_twitter_sentiment_agent.py   - twitter agent
# a2a_iphone_sentiment_agent.py    - Iphone agent

# 

# Pseudocode 
A user submits a sentiment query through the MCP Client.

The MCP Client forwards this query to the central MCP Server.

The Routing Agent classifies whether the input is related to iPhone reviews or Twitter posts using keyword or embedding-based classification.

Depending on the classification, the Routing Agent delegates the request to the appropriate agent via A2A:

iPhone queries are passed to the iPhone Sentiment Agent.

Twitter queries are passed to the Twitter Sentiment Agent.

The agents process the text using their respective ML models:

iPhone agent uses TF-IDF + Random Forest or TextBlob

Twitter agent uses fine-tuned BERTweet transformer

The result is sent back to the Routing Agent, and finally returned to the user through the MCP communication loop.


iPhone Sentiment Agent (invoked via A2A)

Twitter Sentiment Agent (invoked via A2A)

Each agent may interact with either a traditional ML model (iPhone) or a transformer model (Twitter).

The final results are routed via the MCP Client and returned to the user via the MCP Server.



#
# On terminal 1

**Step 1: Create the virtual environment**

python3 -m venv a2a-venv

**Step 2: Activate the environment**
On macOS/Linux:

source a2a-venv/bin/activate

pip install --upgrade pip 

pip install transformers sentence-transformers pandas matplotlib seaborn 

pip install -e . python 

**Step 3: Activate the environment**

python ./test_a2a_sentiment_with_plot.py
