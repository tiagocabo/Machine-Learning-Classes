import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# Example dataset (replace this with your own data reading logic)
reviews = [
    ("The product is amazing, but the delivery was terrible.", 2),  # Mixed sentiment
    ("I love the quality and design!", 1),  # Positive sentiment
    ("The packaging was awful, but the product itself is decent.", 2),  # Mixed sentiment
    ("Completely disappointed, it broke after one use.", 0),  # Negative sentiment
    ("Great value for the price!", 1),  # Positive sentiment
    ("Not what I expected. The build quality is poor.", 0),  # Negative sentiment
    ("It's okay, not great but not terrible either.", 2),  # Neutral sentiment
]

# Extract sentences and labels
sentences = [review for review, _ in reviews]
labels = [label for _, label in reviews]

# TFIDF 
# 2. TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=20)  # You can increase max_features for more complex datasets
X = vectorizer.fit_transform(sentences).toarray()

# Convert labels to tensors
y = torch.tensor(labels, dtype=torch.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # Features should be float32
X_test = torch.tensor(X_test, dtype=torch.float32)

# Data Loader
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y  # Labels are 1D tensors here

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# DNN
class SentimentDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(SentimentDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1]
model = SentimentDNN(input_dim)


# Trainning
# Hyperparameters and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for words, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(words)
            print(outputs, labels)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += (outputs.argmax(1) == labels).sum().item()
        
        accuracy = epoch_acc / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f} - Accuracy: {accuracy:.4f}")

train_model(model, train_loader)

# evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for words, labels in test_loader:
            outputs = model(words)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate_model(model, test_loader)


# inference on new data
def predict_sentiment(model, review):
    model.eval()
    with torch.no_grad():
        review_vectorized = vectorizer.transform([review]).toarray()
        review_tensor = torch.tensor(review_vectorized, dtype=torch.float32)
        output = model(review_tensor)
        prediction = output.argmax(1).item()
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral/Mixed"}
        return sentiment_map[prediction]

new_review = "The product quality is great, but the shipping was slow."
predicted_sentiment = predict_sentiment(model, new_review)
print(f"Review: {new_review}")
print(f"Predicted Sentiment: {predicted_sentiment}")
