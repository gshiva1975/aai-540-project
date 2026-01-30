Execution Output

(a2a-venv) gshiva@EXT-QHW57CK6MK a2a_sentiment_fixed % python ./test_a2a_sentiment_with_plot.py
/Users/gshiva/Downloads/USD-P1/AAI-510-assignment-/project/a2a-venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
[A2A Agent: coordinator_agent] Initialized
Device set to use mps:0
[A2A Agent: iphone_sentiment] Initialized
Device set to use mps:0
[iphone_sentiment] Loaded 3062 iPhone entries
[A2A Agent: iphone_sentiment] Initialized
Device set to use mps:0
[iphone_sentiment] Loaded 3062 iPhone entries

🧪 Prompt: The iPhone battery drains too fast.
[Routing] 'The iPhone battery drains too fast.' → iphone_sentiment (label='iPhone-related issues or praise', score=0.92)
[coordinator_agent] Calling tool: analyze_prompt on agent: iphone_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
[Gangadhar main] {'text': 'Most similar user review: "B09G9BQS98"', 'sentiment': 'NEGATIVE (0.923)'}
[coordinator_agent] Sending response: [✓ iphone_sentiment] NEGATIVE (0.923): Most similar user review: "B09G9BQS98"

🧪 Prompt: The new iPhone design is stunning!
[Routing] 'The new iPhone design is stunning!' → iphone_sentiment (label='iPhone-related issues or praise', score=0.96)
[coordinator_agent] Calling tool: analyze_prompt on agent: iphone_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
[Gangadhar main] {'text': 'Most similar user review: "B09G9HRYFZ"', 'sentiment': 'POSITIVE (0.989)'}
[coordinator_agent] Sending response: [✓ iphone_sentiment] POSITIVE (0.989): Most similar user review: "B09G9HRYFZ"

🧪 Prompt: Twitter keeps crashing on my phone.
[Routing] 'Twitter keeps crashing on my phone.' → twitter_sentiment (label='Twitter-related experiences or comments', score=0.93)
[coordinator_agent] Calling tool: analyze_prompt on agent: twitter_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
[Gangadhar main] {'text': 'Most similar review: "B09V4MXBSN"', 'sentiment': 'NEGATIVE'}
[coordinator_agent] Sending response: [✓ twitter_sentiment] NEGATIVE: Most similar review: "B09V4MXBSN"

🧪 Prompt: I had an amazing time using Twitter Spaces.
[Routing] 'I had an amazing time using Twitter Spaces.' → twitter_sentiment (label='Twitter-related experiences or comments', score=0.96)
[coordinator_agent] Calling tool: analyze_prompt on agent: twitter_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
[Gangadhar main] {'text': 'Most similar review: "B09G9HRYFZ"', 'sentiment': 'POSITIVE'}
[coordinator_agent] Sending response: [✓ twitter_sentiment] POSITIVE: Most similar review: "B09G9HRYFZ"

🧪 Prompt: Why are people unhappy with the new iPhone?
[Routing] 'Why are people unhappy with the new iPhone?' → iphone_sentiment (label='iPhone-related issues or praise', score=0.76)
[coordinator_agent] Calling tool: analyze_prompt on agent: iphone_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
[Gangadhar main] {'text': 'Most similar user review: "B09G9BQS98"', 'sentiment': 'NEGATIVE (0.867)'}
[coordinator_agent] Sending response: [✓ iphone_sentiment] NEGATIVE (0.867): Most similar user review: "B09G9BQS98"

🧪 Prompt: Twitter is full of trolls lately.
[Routing] 'Twitter is full of trolls lately.' → twitter_sentiment (label='Twitter-related experiences or comments', score=1.00)
[coordinator_agent] Calling tool: analyze_prompt on agent: twitter_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
[Gangadhar main] {'text': 'Most similar review: "B09V4MXBSN"', 'sentiment': 'NEGATIVE'}
[coordinator_agent] Sending response: [✓ twitter_sentiment] NEGATIVE: Most similar review: "B09V4MXBSN"

🧪 Prompt: Are iPhone users complaining on Twitter?
[Routing] 'Are iPhone users complaining on Twitter?' → twitter_sentiment (label='Twitter-related experiences or comments', score=0.93)
[coordinator_agent] Calling tool: analyze_prompt on agent: twitter_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
[Gangadhar main] {'text': 'Most similar review: "B09G9BQS98"', 'sentiment': 'NEUTRAL'}
[coordinator_agent] Sending response: [✓ twitter_sentiment] NEUTRAL: Most similar review: "B09G9BQS98"

🧪 Prompt: I saw a tweet saying iPhones overheat easily.
[Routing] 'I saw a tweet saying iPhones overheat easily.' → twitter_sentiment (label='Twitter-related experiences or comments', score=0.81)
[coordinator_agent] Calling tool: analyze_prompt on agent: twitter_sentiment
[iphone_sentiment] Calling tool: analyze_prompt
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
[Gangadhar main] {'text': 'Most similar review: "B09G9HRYFZ"', 'sentiment': 'NEGATIVE'}
[coordinator_agent] Sending response: [✓ twitter_sentiment] NEGATIVE: Most similar review: "B09G9HRYFZ"
Device set to use mps:0
[A2A Agent: iphone_sentiment] Initialized
Device set to use mps:0
[iphone_sentiment] Loaded 3062 iPhone entries
[A2A Agent: iphone_sentiment] Initialized
Device set to use mps:0
[iphone_sentiment] Loaded 3062 iPhone entries

--- Routing Accuracy ---
Routing Accuracy: 2/2 = 1.00

--- Evaluating iPhone Sentiment Agent ---
\n--- Sentiment Classification Metrics ---
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
ROC-AUC Score: Skipped (only one class in y_true)

--- Evaluating Twitter Sentiment Agent ---
 twitter Gangadhar [iphone_sentiment] analyze_prompt called!
\n--- Sentiment Classification Metrics ---
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0
ROC-AUC Score: Skipped (only one class in y_true)
(a2a-venv) gshiva@EXT-QHW57CK6MK a2a_sentiment_fixed %
