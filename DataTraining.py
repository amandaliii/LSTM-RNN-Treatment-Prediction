from DataProcessing import load_mimic3_data
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"

result = load_mimic3_data(mimic_data_dir, nrows=100000)

# Flatten all sequences across all HADM_IDs into one list of category sequences
def extract_sequences_with_hadm_ids(data):
    # returns a flat list of (hadm_id, category, sequence) tuples
    sequence_list = []
    for hadm_id, category_dict in data.items():
        for category, items in category_dict.items():
            if len(items) >= 2:
                sequence_list.append((hadm_id, category, items))
    return sequence_list

# map categorical codes to tokens
def build_vocab(sequences):
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    return item2idx, idx2item

# dataset generator
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        self.pairs = []
        self.max_len = max_len
        self.item2idx = item2idx

        for seq in sequences:
            # convert items to idx, truncate or pad as needed
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]  # sliding window
                target = idx_seq[i]
                self.pairs.append((input_seq, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        return torch.tensor(input_seq), torch.tensor(target)

# model definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        out = self.fc(h_n[-1])
        return out

# train the model
def train_model(model, dataloader, epochs=5, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# predict the next term
def predict_next(model, input_seq, item2idx, idx2item, max_len=20):
    model.eval()
    device = next(model.parameters()).device
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq][-max_len:]
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids
    input_tensor = torch.tensor([input_ids]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_id = logits.argmax(dim=-1).item()
        return idx2item[pred_id]

# still use a single vocabulary for all categories
sequence_tuples = extract_sequences_with_hadm_ids(result)
all_sequences = [seq for _, _, seq in sequence_tuples]
item2idx, idx2item = build_vocab(all_sequences)
dataset = SequenceDataset(all_sequences, item2idx)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMModel(vocab_size=len(item2idx), embed_size=64, hidden_size=128)
train_model(model, loader, epochs=5)

# Try prediction
print("\n=== HADM_ID + Category-wise Predictions ===\n")

N = 3  # number of examples to show

# Shuffle to show variety
random.shuffle(sequence_tuples)

for hadm_id, categories_dict in result.items():
    print(f"\nPredictions for HADM_ID: {hadm_id}")
    for category, items in categories_dict.items():
        if len(items) < 2:
            continue  # Not enough data to predict
        # Use last N items as input (e.g. N=5, or all but last if you want to leave one for 'ground truth')
        input_seq = items[:5]  # or items[:-1] if predicting last item
        predicted_next = predict_next(model, input_seq, item2idx, idx2item)
        print(f"  Category: {category}")
        print(f"    Input sequence: {input_seq}")
        print(f"    Predicted next item: {predicted_next}")
