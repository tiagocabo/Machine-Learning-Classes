
# JOB descriptions dataset
job_descriptions = [
    ("We are looking for a Senior Data Scientist with 5+ years of experience.", ["O", "O", "O", "O", "B-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O", "O"]),
    ("The company is hiring a Junior Software Engineer to work on cloud projects.", ["O", "O", "O", "O", "B-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O", "O", "O"]),
    ("Looking for a Project Manager to oversee multiple client engagements.", ["O", "O", "O", "B-TITLE", "I-TITLE", "O", "O", "O", "O", "O", "O"]),
    ("The Marketing Specialist will drive our campaign strategy.", ["O", "B-TITLE", "I-TITLE", "O", "O", "O", "O", "O"]),
]

# Data preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Create a vocabulary and tag mapping
words = list(set(word for sentence, _ in job_descriptions for word in sentence.split()))
tags = list(set(tag for _, label in job_descriptions for tag in label))

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1  # Unknown words
word2idx["PAD"] = 0  # Padding

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Convert sentences and labels to sequences
X = [[word2idx.get(w, word2idx["UNK"]) for w in sentence.split()] for sentence, _ in job_descriptions]
y = [[tag2idx[t] for t in label] for _, label in job_descriptions]

# Padding sequences
max_len = max(len(s) for s in X)
X = [s + [word2idx["PAD"]] * (max_len - len(s)) for s in X]
y = [l + [tag2idx["O"]] * (max_len - len(l)) for l in y]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Torch data loader
class JobDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = JobDataset(X_train, y_train)
test_dataset = JobDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# BiLSTM Model

import torch.nn as nn
import torch.optim as optim

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=100):
        super(BiLSTM_NER, self).__init__()
        
        # The next line could change case we want to use word2vec embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)  # 2 for bidirection
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

vocab_size = len(word2idx)
tagset_size = len(tag2idx)

model = BiLSTM_NER(vocab_size, tagset_size)

# Training 
# Hyperparameters and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["PAD"])  # Ignore padding

# Training Loop

def train_model(model, train_loader, epochs=10):
    # change to train mode
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for words, labels in train_loader:
            # init gradient
            optimizer.zero_grad()
            outputs = model(words)
            outputs = outputs.view(-1, outputs.shape[-1])  # Flatten to [batch_size * seq_len, num_classes]
            labels = labels.view(-1)  # Flatten labels to match
            loss = criterion(outputs, labels)
            
            # back propagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

train_model(model, train_loader)

# Model evaluation 
def evaluate_model(model, test_loader):
    # change to inference mode
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for words, labels in test_loader:
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=-1)
            mask = labels != word2idx["PAD"]  # Ignore padding tokens
            correct += torch.sum((predicted == labels) * mask).item()
            total += torch.sum(mask).item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

evaluate_model(model, test_loader)

# Inference on new examples
def predict_job_title(model, sentence):
    model.eval()
    with torch.no_grad():
        sentence_idx = [word2idx.get(w, word2idx["UNK"]) for w in sentence.split()]
        padded_sentence = sentence_idx + [word2idx["PAD"]] * (max_len - len(sentence_idx))
        input_tensor = torch.tensor([padded_sentence], dtype=torch.long)
        output = model(input_tensor)
        _, predicted = torch.max(output, dim=-1)
        return [idx2tag[idx.item()] for idx in predicted[0] if idx.item() != word2idx["PAD"]]

test_sentence = "We are seeking a Data Analyst for our team."
predicted_tags = predict_job_title(model, test_sentence)
print("Sentence:", test_sentence.split())
print("Predicted Tags:", predicted_tags)

