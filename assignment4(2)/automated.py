import ollama
import random

# Ground truth sentiment (optional, if you want scoring)
true_sentiment = "positive"
test_text = "I love this product, it works great!"

# Prompt templates to explore automatically
prompt_templates = [
    """
Analyze the sentiment of the sentence. Choose from 'positive', 'neutral', or 'negative'.

Examples:
1. Text: "I hate waiting in line."
   Sentiment: negative

2. Text: "The food was okay."
   Sentiment: neutral

3. Text: "I absolutely loved the experience!"
   Sentiment: positive

Text: "{text}"
Sentiment:""",
    
    """
Determine sentiment. Options: positive / neutral / negative.

Input: "{text}"
Sentiment:""",
    
    """
Classify the sentiment of the following text with one word only.

"{text}"
Sentiment:""",
    
    """
Please provide the sentiment of this statement: "{text}".
Return just one word: positive, neutral, or negative.
Sentiment:""",
    
    """
Text: "{text}"
What is the sentiment? (Only respond with: positive / neutral / negative)
Sentiment:"""
]

def monte_carlo_prompt_search(text, n_trials=5):
    tried_prompts = random.sample(prompt_templates, k=n_trials)
    results = []

    for prompt_template in tried_prompts:
        prompt = prompt_template.format(text=text).strip()
        response = ollama.generate(
            model="mistral",
            prompt=prompt
        )
        results.append({
            "prompt": prompt,
            "response": response['response'].strip()
        })

    return results

# Run Monte Carlo search
results = monte_carlo_prompt_search(test_text, n_trials=5)

# Print results
for idx, result in enumerate(results):
    print(f"\n--- Variation {idx+1} ---")
    print("Prompt:\n", result["prompt"])
    print("Response:", result["response"])
