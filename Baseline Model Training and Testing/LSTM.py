import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re
import string
import gensim.downloader as api

# ---------- Load CSVs ----------
train = pd.read_csv('sms_train.csv')
val = pd.read_csv('sms_val.csv')
test = pd.read_csv('sms_test.csv')

# ---------- Label mapping ----------
label_map = {'ham': 0, 'spam': 1}
for df in [train, val, test]:
    df['label'] = df['Category'].map(label_map)

# ---------- Text preprocessing ----------
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

for df in [train, val, test]:
    df['Message'] = df['Message'].apply(preprocess)

# ---------- Tokenisation ----------
def tokenize(text):
    return text.split()

# ---------- Build vocab ----------
counter = Counter()
for msg in train['Message']:
    counter.update(tokenize(msg))
vocab = {word for word, freq in counter.items() if freq >= 2}

PAD_IDX = 0
UNK_IDX = 1
itos = ['<PAD>', '<UNK>'] + sorted(vocab)
stoi = {word: idx for idx, word in enumerate(itos)}

def numericalize(text):
    return [stoi.get(token, UNK_IDX) for token in tokenize(text)]

# ---------- Dataset ----------
class SMSDataset(Dataset):
    def __init__(self, df):
        self.texts = df['Message'].tolist()
        self.labels = df['label'].tolist()
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(numericalize(self.texts[idx]), dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded_texts, labels

BATCH_SIZE = 32
train_loader = DataLoader(SMSDataset(train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(SMSDataset(val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(SMSDataset(test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------- Load pretrained GloVe embeddings ----------
print("Loading GloVe...")
glove_model = api.load("glove-wiki-gigaword-100")
embed_dim = 100
embedding_matrix = np.zeros((len(itos), embed_dim))
for idx, word in enumerate(itos):
    if word in glove_model:
        embedding_matrix[idx] = glove_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

# ---------- Model ----------
class LSTMSMSClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False  # freeze embeddings for stability
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # concat last layers' hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMSMSClassifier(len(itos), embed_dim).to(device)

# ---------- Training setup ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
MAX_EPOCHS = 20
PATIENCE = 3
CLIP = 1.0  # gradient clipping

# ---------- Training / Eval functions ----------
def train_epoch(loader):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item() * texts.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return epoch_loss/total, correct/total

def eval_epoch(loader):
    model.eval()
    epoch_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = model(texts)
            loss = criterion(preds, labels)
            epoch_loss += loss.item() * texts.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return epoch_loss/total, correct/total, all_preds, all_labels

# ---------- Train with early stopping ----------
best_val_acc = 0
patience_counter = 0
for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = train_epoch(train_loader)
    val_loss, val_acc, _, _ = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# ---------- Load best model and test ----------
model.load_state_dict(best_state)
test_loss, test_acc, test_preds, test_labels = eval_epoch(test_loader)
print(f"\nTest Accuracy: {test_acc:.4f}")
print("\nClassification Report (Test):")
print(classification_report(test_labels, test_preds, target_names=['ham', 'spam']))
print("Confusion Matrix (Test):")
print(confusion_matrix(test_labels, test_preds))
