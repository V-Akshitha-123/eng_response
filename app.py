
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from flask import Flask ,request, jsonify,send_file,after_this_request
from collections import Counter
from flask_cors import CORS
from gtts import gTTS

import uuid
import os
# Load Dataset
url = f"https://drive.google.com/uc?id=1RCZShB5ohy1HdU-mogcP16TbeVv9txpY"
df = pd.read_csv(url)
df = df.dropna(subset=['query', 'response'])

# Ensure all entries are strings
df['query'] = df['query'].astype(str)
df['response'] = df['response'].astype(str)
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
tokenizer.build_vocab(train_data["query"].tolist() + train_data["response"].tolist())

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]["query"]
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
        output = self.transformer(tgt_emb.permute(1, 0, 2), src_emb.permute(1, 0, 2), tgt_mask=tgt_mask)
        return self.fc_out(output.permute(1, 0, 2))

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(tokenizer.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

def load_model(model, path="gpt_model.pth"):
    model.load_state_dict(torch.load(path, map_location=device,weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded from", path)

load_model(model)

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

app=Flask(__name__)
CORS(app)

@app.route("/intent")
def home():
    return jsonify({"intents" :list(set(df['intent'].dropna()))})

@app.route("/query", methods=["POST"])
def query_model():
    global audio_telugu_response
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Assuming `generate_response` is a function that processes the query
    response = generate_response(model, query)
    print(response)
    
    return jsonify({"telugu":(response)})
@app.route("/audio", methods=["POST"])
def get_audio():
    data = request.get_json()
    text = data.get("text")

    # text=audio_telugu_response
    if not text:
        return jsonify({"error": "No Response To convert to speech"}), 400

    filename = f"speech_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join("audio_temp", filename)

    os.makedirs("audio_temp", exist_ok=True)

    # Convert text to Telugu speech
    speech = gTTS(text=text, lang="en")
    speech.save(filepath)

    # Automatically delete the file after sending
    @after_this_request
    def cleanup(response):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Cleanup error: {e}")
        return response

    return send_file(filepath, mimetype="audio/mpeg", as_attachment=False)

