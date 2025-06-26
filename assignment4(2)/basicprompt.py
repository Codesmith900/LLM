#Zero-shot learning

import ollama

response = ollama.generate(
    model="mistral",  # or any other model you prefer
    prompt="Determine if the sentiment of this text is positive, neutral, or negative: 'I love this product, it works great!'"
)

print(response)

#gives basic output : positive