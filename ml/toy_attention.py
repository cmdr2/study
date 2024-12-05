import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
import safetensors.torch
import os
import json


MODEL_FILE = "char_model.sft"
VOCAB_FILE = "char_model_vocab.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
embed_size = 16
batch_size = 512
num_heads = 4
epochs = 2000
lr = 1e-3
max_len = 8

# Seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TextDataset(Dataset):
    def __init__(self):
        self.dataset = None
        self.vocab = []

        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        if os.path.exists(VOCAB_FILE):
            self._load_vocab()
            self._process_vocab()
        else:
            self.load_dataset()

    def load_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset("community-datasets/generics_kb", split="train[:1000]")
            print("loaded dataset")

            # Build vocabulary based on characters
            self.vocab = set()
            for sentence in self.dataset["generic_sentence"]:
                self.vocab.update(sentence)  # Each character is part of the vocabulary

            self.vocab = list(self.vocab)
            self._process_vocab()
            self._save_vocab()

    def _process_vocab(self):
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)

        print(f"Vocab ({len(self.vocab)}): {self.vocab}")

    def _save_vocab(self):
        with open(VOCAB_FILE, "w") as f:
            json.dump(self.vocab, f)

    def _load_vocab(self):
        with open(VOCAB_FILE, "r") as f:
            self.vocab = json.load(f)

    def tokenize(self, sentence):
        return [self.char_to_idx.get(char, 0) for char in sentence[:max_len]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset["generic_sentence"][idx][:max_len]
        input_sequence = sentence[:-1]  # All but the last character
        target_char = sentence[-1] if len(sentence) > 1 else "<PAD>"
        input_tokens = [self.char_to_idx.get(char, 0) for char in input_sequence]
        target_token = self.char_to_idx.get(target_char, 0)  # The next character (target)
        return torch.tensor(input_tokens), torch.tensor(target_token)


# MultiHeadAttention with masked attention logic
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("mask", torch.tril(torch.ones(1, 1, max_len, max_len)) == 1)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        mask = self.mask[:, :, : scores.size(2), : scores.size(3)].to(scores.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output, attn_weights


# Model for next-character prediction
class PredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(PredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = MultiHeadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        attn_output, attn_weights = self.attention(embeds, embeds, embeds)
        pooled_output = attn_output.mean(dim=1)
        logits = self.fc(pooled_output)
        return logits, attn_weights


# Load the model
def load_model():
    if os.path.exists(MODEL_FILE):
        sd = safetensors.torch.load_file(MODEL_FILE)
        model.load_state_dict(sd)
        print("loaded model")
        del sd


# Training Loop
def train():
    # Custom collate function to pad sequences
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        padded_inputs = [F.pad(input, (0, max_len - len(input)), value=0) for input in inputs]
        return torch.stack(padded_inputs, dim=0), torch.stack(targets, dim=0)

    dataset.load_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def train_model(model, dataloader, optimizer, criterion, device, progress_bar):
        model.train()
        total_loss = 0
        for inputs, target in dataloader:
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)  # Only use logits for loss computation
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())

        return total_loss / len(dataloader)

    for epoch in range(epochs):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch: {epoch}")
        avg_loss = train_model(model, dataloader, optimizer, criterion, device, progress_bar)


# Save the model
def save_model():
    safetensors.torch.save_file(model.state_dict(), MODEL_FILE)


def evaluate_model():
    # Evaluation Functions
    def predict_next_char(text):
        tokenized_input = torch.tensor([dataset.tokenize(text)], dtype=torch.long).to(device)
        with torch.no_grad():
            model.eval()
            logits, attn_weights = model(tokenized_input)
            predicted_token = torch.argmax(logits, dim=-1).item()
            predicted_char = dataset.idx_to_char.get(predicted_token, "<UNK>")
            # print("Attention Weights:", attn_weights)
            return predicted_char

    def generate_text(start_text, length):
        generated_text = start_text
        for _ in range(length):
            next_char = predict_next_char(generated_text)
            generated_text += next_char
        return generated_text

    start_text = "This is"
    print("Start Text:", start_text)
    print("Generated Text:", start_text + predict_next_char(start_text))

    print("Another generated Text:", generate_text("", 50))


# Main
dataset = TextDataset()
model = PredictionModel(vocab_size=dataset.vocab_size, embed_size=embed_size).to(device)

if __name__ == "__main__":
    load_model()
    # train()
    # save_model()
    evaluate_model()
