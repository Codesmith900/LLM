"""Let's improve the prompt manually with these techniques:

    Adding clear instructions

    Including examples (few-shot learning)

    Specifying output format

"""
import ollama

improved_prompt = """
Perform sentiment analysis on the given text. Classify the sentiment as either 'positive', 'neutral', or 'negative'.

Examples:
1. Text: "This is the best day ever!"
   Sentiment: positive

2. Text: "The package arrived on time."
   Sentiment: neutral

3. Text: "I'm really disappointed with the service."
   Sentiment: negative

Now analyze this text:
Text: "I love this product, it works great!"
Sentiment:"""

response = ollama.generate(
    model="mistral",
    prompt=improved_prompt
)

print(response)