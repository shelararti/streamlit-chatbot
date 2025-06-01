import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Train chatbot model")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_path', type=str, default='data.pth')
args = parser.parse_args()

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

def load_intents(file_path='intents.json'):
    with open(file_path, 'r', encoding='utf-8') as f:  # <- Set encoding
        return json.load(f)


def preprocess_data(intents):
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X = []
    y = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X.append(bag)
        label = tags.index(tag)
        y.append(label)

    return np.array(X), np.array(y), all_words, tags

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y
        self.n_samples = len(X)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def evaluate(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val).to(device)
        labels = torch.tensor(y_val).to(device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        acc = (predicted == labels).sum().item() / len(y_val)
    return acc

def train():
    intents = load_intents()
    X, y, all_words, tags = preprocess_data(intents)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    input_size = X.shape[1]
    output_size = len(tags)
    hidden_size = 8
    print(f"[INFO] Input size: {input_size}, Output size: {output_size}")

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            acc = evaluate(model, X_val, y_val)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item():.4f}, Val Accuracy: {acc:.2f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "model_state": model.state_dict(),
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "all_words": all_words,
                "tags": tags
            }, args.save_path)

    print(f"[INFO] Training complete. Best loss: {best_loss:.4f}")
    print(f"[INFO] Model saved to {args.save_path}")

if __name__ == "__main__":
    train()