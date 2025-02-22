import time
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from tqdm import tqdm

client = OpenAI(api_key=api_key)

# Load the dataset
ds = load_dataset("mteb/tweet_sentiment_extraction")
test_texts = ds["test"]["text"]
true_labels = ds["test"]["label"]  # Assuming this is the true sentiment label


# Define a function to get sentiment classification from the OpenAI API
def classify_sentiment(text, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user",
             "content": f"classify the sentiment of this text: '{text}'. Return ONLY 1 word from the list ['positive', 'negative', 'neutral']."}
        ]
    )
    sentiment = response.choices[0].message.content.strip().lower()
    return sentiment

models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"]
models = ["gpt-3.5-turbo", "gpt-4o-mini"]

for model in models:
    tic = time.time()
    # Classify sentiments for a subset to avoid rate limits
    predicted_labels = [classify_sentiment(text, model=model) for text in tqdm(test_texts)]  # Limiting to 50 for demonstration
    toc = time.time()

    true_labels = true_labels[:50]  # Match the subset size


    # map to integer
    mapper = {"negative":0, "neutral":1, "positive":2}
    predicted_labels_mapped = list(map(mapper.get, predicted_labels))

    # Evaluate the results
    print(f"model {model} took: {toc - tic} s")
    print(f" {model} Accuracy:", accuracy_score(true_labels, predicted_labels_mapped))
    print(f" {model} Recall:", recall_score(true_labels, predicted_labels_mapped, average="macro"))
    print(f" {model} F1:", f1_score(true_labels, predicted_labels_mapped, average="macro"))

    #print("\n Classification Report:")
    #print(classification_report(true_labels, predicted_labels_mapped))

