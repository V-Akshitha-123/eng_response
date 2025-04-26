import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
import torch
import os

PORT=7001
url = f"https://drive.google.com/uc?id=1RCZShB5ohy1HdU-mogcP16TbeVv9txpY"
df = pd.read_csv(url)


# Tokenizer (Scratch)
class ScratchTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, max_len=200):
        tokens = [self.word2idx.get(word, 3) for word in text.split()]
        tokens = [1] + tokens[:max_len - 2] + [2]
        return tokens + [0] * (max_len - len(tokens))

    def decode(self, tokens):
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in tokens if idx > 0])

# Train-Test Split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Initialize Tokenizer
tokenizer = ScratchTokenizer()
tokenizer.build_vocab(train_data["instruction"].tolist() + train_data["response"].tolist())

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]["instruction"]
        tgt_text = self.data.iloc[idx]["response"]
        src = torch.tensor(self.tokenizer.encode(src_text), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(tgt_text), dtype=torch.long)
        return src, tgt

# Load Dataset
train_dataset = TextDataset(train_data, tokenizer)
test_dataset = TextDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Improved GPT-Style Transformer Model

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=8, num_layers=6, max_len=200):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_size))
        # The problem was here, setting num_encoder_layers to 0
        # makes the model try to access a non-existent layer.
        # The solution is to remove the encoder completely.
        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads), num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_embedding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_embedding[:, :tgt.size(1), :]

        # Causal Mask for Auto-Regressive Decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Since we are using only the decoder now,
        # we need to pass the source embeddings as memory.
        output = self.transformer(tgt_emb.permute(1, 0, 2), src_emb.permute(1, 0, 2), tgt_mask=tgt_mask)
        return self.fc_out(output.permute(1, 0, 2))

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(tokenizer.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Load the model
def load_model(model, path="gpt_model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Model file not found!")

load_model(model)

# Training Function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



# Generate Response
def generate_response(model, query, max_length=200):
    model.eval()
    src = torch.tensor(tokenizer.encode(query)).unsqueeze(0).to(device)
    tgt = torch.tensor([[1]]).to(device)  # <SOS>

    for _ in range(max_length):
        output = model(src, tgt)
        next_word = output.argmax(-1)[:, -1].unsqueeze(1)
        tgt = torch.cat([tgt, next_word], dim=1)
        if next_word.item() == 2:  # <EOS>
            break

    return tokenizer.decode(tgt.squeeze(0).tolist())
print(set(df['intent']))

# Test Query
app = Flask(__name__)

@app.route("/")
def home():
    return "Transformer-based Response Generator API is running!"

@app.route("/query", methods=["POST"])
def query_model():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    response = generate_response(model, query)
    return jsonify({"query": query, "response": response})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001)
