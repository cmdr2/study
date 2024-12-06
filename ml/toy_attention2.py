import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
import safetensors.torch
import os


MODEL_FILE = "trained_models/classify_starlink_model.sft"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameters
embed_size = 6
batch_size = 64
num_heads = 1
epochs = 300
lr = 1e-3
max_len = 6

# Seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

POSITIVE_PHRASES = ["saw", "could see", "watched"]
NEGATIVE_PHRASES = ["did not see", "could not see"]


class TestDataset(Dataset):
    def __init__(self, num_examples=100):
        self.vocab = ["UNKNOWN", "SUCCESS", "FAIL", "i", "starlink"]
        for phrases in (POSITIVE_PHRASES, NEGATIVE_PHRASES):
            for phrase in phrases:
                for w in phrase.split():
                    if w not in self.vocab:
                        self.vocab.append(w)

        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        self.vocab_size = len(self.vocab)
        self.num_examples = num_examples

        print(f"Vocab ({self.vocab_size}): {self.vocab}")

    def tokenize(self, sentence):
        return [self.token_to_idx.get(token, 0) for token in sentence.split()]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        label = "SUCCESS" if random.random() >= 0.5 else "FAIL"
        phrase_book = POSITIVE_PHRASES if label == "SUCCESS" else NEGATIVE_PHRASES

        sentence = f"i {random.choice(phrase_book)} starlink"

        input_tokens = self.tokenize(sentence)
        target_token = self.tokenize(label)[0]  # The next token (target)
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

        # self.register_buffer("mask", torch.tril(torch.ones(1, 1, max_len, max_len)) == 1)

    def forward(self, query, key, value, print_actual_output=False):
        batch_size = query.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if print_actual_output:
            print("query:", query.shape)
            print(query)
            print("key:", key.shape)
            print(key)
            print("value:", value.shape)
            print(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        # mask = self.mask[:, :, : scores.size(2), : scores.size(3)].to(scores.device)
        # scores = scores.masked_fill(mask == 0, float("-inf"))

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

    def forward(self, x, input_length, print_actual_output=False):
        embeds = self.embedding(x)
        attn_output, attn_weights = self.attention(embeds, embeds, embeds, print_actual_output)

        # mask to zero out the contribution from the padding tokens
        mask = torch.arange(attn_output.size(1)).unsqueeze(0).to(device) < input_length.unsqueeze(-1)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, attn_output.size(-1)).to(device)
        actual_attn_output = attn_output * expanded_mask
        if print_actual_output:
            print("actual_output:")
            print(actual_attn_output)

        pooled_output = actual_attn_output.mean(dim=1)
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
        input_lengths = [torch.tensor(len(input)) for input in inputs]
        return torch.stack(padded_inputs, dim=0), torch.stack(input_lengths), torch.stack(targets, dim=0)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def train_model(model, dataloader, optimizer, criterion, device, progress_bar):
        model.train()
        total_loss = 0
        for inputs, input_lengths, target in dataloader:
            inputs, input_lengths, target = inputs.to(device), input_lengths.to(device), target.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs, input_lengths)  # Only use logits for loss computation
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
    def predict_next_word(text, show_partial=True):
        tokenized_input = torch.tensor(dataset.tokenize(text)).unsqueeze(0)
        padded_inputs = [F.pad(input, (0, max_len - len(input)), value=0) for input in tokenized_input]
        padded_inputs = torch.stack(padded_inputs).to(device)
        input_lengths = [torch.tensor(len(input)) for input in tokenized_input]
        input_lengths = torch.stack(input_lengths).to(device)
        with torch.no_grad():
            model.eval()
            logits, attn_weights = model(padded_inputs, input_lengths, print_actual_output=True)
            predicted_token = torch.argmax(logits, dim=-1).item()
            predicted_char = dataset.idx_to_token.get(predicted_token, "<UNK>")
            print("Attention Weights:")
            x = attn_weights.squeeze().squeeze().cpu().numpy()
            n = len(tokenized_input[0])
            rows = x[:n] if show_partial else x
            for row in rows:
                row = row[:n] if show_partial else row
                row = [f"{y:0.1f}" for y in row]
                print(row)
            return predicted_char

    for text in (
        "i saw starlink",
        "i did not see starlink",
        "i could not see starlink",
        "i could see starlink",
        "i watched starlink",
        "i saw",
        "i not",
        "not",
        "saw",
        "starlink",
    ):
        print("Input:", text)
        print("Label:", predict_next_word(text, show_partial=False))

        print("\n\n")


# Main
dataset = TestDataset()
model = PredictionModel(vocab_size=dataset.vocab_size, embed_size=embed_size).to(device)

if __name__ == "__main__":
    load_model()
    train()
    save_model()
    evaluate_model()
