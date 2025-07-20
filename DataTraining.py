# import dataset from dataprocessing.py
from DataProcessing import load_mimic3_data
# used to count frequency of items (for building vocabulary) 
from collections import Counter
# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# used for shuffling 
import random
# stores and exports predictions into excel file 
import pandas as pd

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=500000)

# Flatten all sequences across all HADM_IDs into one list of category sequences
def extract_sequences_with_hadm_ids(data):
    # returns a flat list of (hadm_id, category, sequence) tuples
    sequence_list = []
    for hadm_id, category_dict in data.items():
        for category, items in category_dict.items():
            # makes sure there's at least two items in each category to be used for prediction
            if len(items) >= 2:
                sequence_list.append((hadm_id, category, items))
    return sequence_list

# map categorical codes to tokens
def build_vocab(sequences):
    # counts occurrence of all items in a sequence
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    # returns the mapping of item to index and index to item 
    return item2idx, idx2item

# dataset generator
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        self.pairs = []
        self.max_len = max_len
        self.item2idx = item2idx

        # converts sequences into (input_seq, target) pair for training 
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

# basic 1-layer LSTM (long short term memory) model definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMModel, self).__init__()
        # embedds item tokens 
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # feeds them through the LSTM and outputs logits (unnormalized predictions) over entire vocabulary
    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        out = self.fc(h_n[-1])
        return out

# train the model for epochs times 
def train_model(model, dataloader, epochs, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # for each batch: forward pass -> loss -> backward pass -> update weights 
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
def predict_next(model, input_seq, item2idx, idx2item, max_len=None):
    model.eval()
    device = next(model.parameters()).device
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]

    # takes an input sequence, pads or trims it to length max_len, and feeds it to the model 
    if max_len is None:
        max_len = len(input_ids)

    input_ids = input_ids[-max_len:]  # Still trim if needed

    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids

    input_tensor = torch.tensor([input_ids]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_id = logits.argmax(dim=-1).item()
        return idx2item[pred_id]

# still use a single vocabulary for all categories
# tuple of sequences, unmodifiable 
sequence_tuples = extract_sequences_with_hadm_ids(result)
# combines all sequences for each category 
all_sequences = [seq for _, _, seq in sequence_tuples]
item2idx, idx2item = build_vocab(all_sequences)
# creates dataset of sequences, mapping the items to the hadm_id 
dataset = SequenceDataset(all_sequences, item2idx)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMModel(vocab_size=len(item2idx), embed_size=64, hidden_size=128)
# runs with epochs 
train_model(model, loader, epochs=5)

# store predictions
prediction_rows = []

# number of examples to show
num_hadms = 20

# number of prediction runs
NUM_RUNS = 5

# run the prediction
print("\n=== HADM_ID + Category-wise Predictions ===\n")
for run in range(1, NUM_RUNS + 1):
    print(f"\nRun {run} Predictions...\n")
    model = LSTMModel(vocab_size=len(item2idx), embed_size=64, hidden_size=128)
    train_model(model, loader, epochs=10)

    # Get all unique HADM_IDs, shuffle, and pick N unique ones
    # filter HADM_IDs to only those with at least one category with ≥ 2 items
    valid_hadm_ids = [
        hadm_id for hadm_id, category_dict in result.items()
        if any(len(items) >= 2 for items in category_dict.values())
    ]

    random.shuffle(valid_hadm_ids)
    hadm_ids_to_process = valid_hadm_ids[:num_hadms]  # use only top N eligible HADM_IDs

    print(f"[Run {run}] Using {len(hadm_ids_to_process)} valid HADM_IDs.")

    for hadm_id in hadm_ids_to_process:
        categories_dict = result[hadm_id]
        print(f"  HADM_ID: {hadm_id}")
        for category, items in categories_dict.items():
            print(f"    Category: {category} → Predicting next item from sequence of length {len(items)}...")
            if len(items) < 2:
                continue
            input_seq = items[:]
            predicted_next = predict_next(model, items, item2idx, idx2item, max_len=len(items))
            prediction_rows.append({
                "Run": run,
                "HADM_ID": hadm_id,
                "Category": category,
                "Input_Sequence": ", ".join(map(str, input_seq)),
                "Predicted_Next_Item": predicted_next
            })

df = pd.DataFrame(prediction_rows)
print("\nTotal predictions stored:", len(prediction_rows))
print(pd.DataFrame(prediction_rows).head(10))  # See example rows

unique_ids = set([row["HADM_ID"] for row in prediction_rows])
print(f"\nTotal unique HADM_IDs predicted across all runs: {len(unique_ids)}")

# saves data to excel file 
df.to_excel("mimic3_predictions.xlsx", index=False)
print("Predictions saved to mimic3_predictions.xlsx")
