import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time 
import math
from transformers import BertTokenizer, BertForSequenceClassification

class FakeNewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): 
        src_padding_mask = (self.sequences[idx] == 0)
        return self.sequences[idx], self.labels[idx], src_padding_mask

class BERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

def conduct_design_studies(vocab_size, loader, device):
    results = {}
    

    print("\n--- Study 1: Positional Encoding Impact ---")
    model_with_pe = TransformerClassifier(vocab_size, d_model=64, nhead=8, use_pe=True)
    model_no_pe = TransformerClassifier(vocab_size, d_model=64, nhead=8, use_pe=False)
    
    results['With PE'] = run_experiment("Transformer (With PE)", model_with_pe, loader, device)
    results['No PE'] = run_experiment("Transformer (No PE)", model_no_pe, loader, device)

   
    print("\n--- Study 2: Attention Heads (2 vs 8) ---")
    model_h2 = TransformerClassifier(vocab_size, d_model=64, nhead=2)
    model_h8 = TransformerClassifier(vocab_size, d_model=64, nhead=8)
    
    results['2 Heads'] = run_experiment("Transformer (2 Heads)", model_h2, loader, device)
    results['8 Heads'] = run_experiment("Transformer (8 Heads)", model_h8, loader, device)
    
    return results

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=8, num_layers=2, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model) if use_pe else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(64)
        if self.use_pe:
            x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        return self.sigmoid(self.fc(output.mean(dim=1))).squeeze()

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, mask=None):
        _, h_n = self.rnn(self.embedding(x))
        return self.sigmoid(self.fc(h_n.squeeze(0))).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, mask=None):
        _, (h_n, _) = self.lstm(self.embedding(x))
        return self.sigmoid(self.fc(h_n.squeeze(0))).squeeze()


def run_experiment(name, model, loader, device, is_bert=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5 if is_bert else 0.001)
    criterion = nn.BCELoss()
    
    
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        if is_bert:
            input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=mask).logits.squeeze()
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
        else:
            texts, labels, masks = [b.to(device) for b in batch]
            outputs = model(texts, masks)
            loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()

    
    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            if is_bert:
                input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = torch.sigmoid(model(input_ids, attention_mask=mask).logits.squeeze())
            else:
                texts, labels, masks = [b.to(device) for b in batch]
                outputs = model(texts, masks)
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    inf_speed = (time.time() - start_time) / len(loader.dataset)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels))*100:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds):.2f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.2f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.2f}")
    print(f"Mean Inf Speed: {inf_speed:.6f}s/sample")
    print(f"Total Parameters: {params:,}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        
        df_true = pd.read_csv('True.csv'); df_true['label'] = 0
        df_fake = pd.read_csv('Fake.csv'); df_fake['label'] = 1
        df = pd.concat([df_true, df_fake]).sample(frac=0.05, random_state=42)

        
        all_text = " ".join(df['text'].astype(str)).lower().split()
        vocab = {word: i+1 for i, word in enumerate(set(all_text[:5000]))}
        encode = lambda t: [vocab.get(w.lower(), 0) for w in str(t).split()][:50]
        X = [seq + [0]*(50-len(seq)) for seq in [encode(t) for t in df['text']]]
        loader = DataLoader(FakeNewsDataset(X, df['label'].values), batch_size=16)

        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encodings = tokenizer(list(df['text'].astype(str)), truncation=True, padding=True, max_length=50)
        bert_loader = DataLoader(BERTDataset(encodings, df['label'].values), batch_size=16)

        
        run_experiment("RNN", VanillaRNN(len(vocab)+1, 32, 64), loader, device)
        run_experiment("LSTM", LSTMModel(len(vocab)+1, 32, 64), loader, device)
        run_experiment("Custom Transformer", TransformerClassifier(len(vocab)+1), loader, device)
        run_experiment("Fine-Tuned BERT", BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1), bert_loader, device, is_bert=True)
        conduct_design_studies(len(vocab)+1, loader, device)

    except Exception as e: print(f"Error: {e}")