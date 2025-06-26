import requests
import json
from time import sleep

class NewsClassifier:
    def __init__(self, model="mistral"):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model
        
    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.ollama_url, json=payload)
        return json.loads(response.text)["response"]
    
    def direct_prompt(self, headline):
        prompt = f"""Classify the following news headline as either "sports" or "business":
        "{headline}"
        """
        return self.generate(prompt)
    
    def few_shot_prompt(self, headline):
        prompt = f"""Here are some examples of news headline classifications:

        Example 1:
        Headline: "Manchester United signs new striker for £50 million"
        Classification: sports

        Example 2:
        Headline: "Federal Reserve raises interest rates by 0.5%"
        Classification: business

        Example 3:
        Headline: "Tesla stock drops after earnings report"
        Classification: business

        Now classify this headline:
        "{headline}"
        """
        return self.generate(prompt)
    
    def chain_of_thought_prompt(self, headline):
        prompt = f"""Let's analyze this news headline step by step to determine if it belongs to "sports" or "business":

        1. Read the headline: "{headline}"
        2. Identify key entities and terms
        3. Consider the context of the headline
        4. Compare with category definitions:
           - Sports: involves teams, players, matches, tournaments
           - Business: involves companies, profits, markets, economy
        5. Analyze which category fits best
        6. Explain your reasoning
        7. Provide final classification

        Final classification analysis:
        """
        return self.generate(prompt)
    
    def evaluate(self, headlines):
        results = []
        for headline, true_label in headlines:
            print(f"\nClassifying: {headline}")
            
            # Get predictions from all three methods
            direct = self.direct_prompt(headline)
            sleep(1)  # Rate limiting
            
            few_shot = self.few_shot_prompt(headline)
            sleep(1)
            
            cot = self.chain_of_thought_prompt(headline)
            sleep(1)
            
            results.append({
                "headline": headline,
                "true_label": true_label,
                "direct": direct,
                "few_shot": few_shot,
                "chain_of_thought": cot
            })
            
            print(f"Direct: {direct}")
            print(f"Few-shot: {few_shot}")
            print(f"Chain-of-thought: {cot}")
        
        return results

if __name__ == "__main__":
    # Test headlines with ground truth labels
    test_headlines = [
        ("Apple unveils new iPhone with revolutionary camera technology", "business"),
        
    ]
    
    classifier = NewsClassifier()
    results = classifier.evaluate(test_headlines)
    
    # Calculate accuracy for each method
    def calculate_accuracy(results, method):
        correct = 0
        for item in results:
            # Simple check if the true label appears in the output
            if item['true_label'].lower() in item[method].lower():
                correct += 1
        return correct / len(results)
    
    print("\nEvaluation Results:")
    print(f"Direct Prompt Accuracy: {calculate_accuracy(results, 'direct')*100:.2f}%")
    print(f"Few-Shot Prompt Accuracy: {calculate_accuracy(results, 'few_shot')*100:.2f}%")
    print(f"Chain-of-Thought Prompt Accuracy: {calculate_accuracy(results, 'chain_of_thought')*100:.2f}%")