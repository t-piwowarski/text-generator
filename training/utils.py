import torch
from torch.utils.data import Dataset

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load text data from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.lower()

# Function to create a character-level vocabulary
def create_vocab(text):
    vocab = sorted(set(text))  # Unique characters in text
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return vocab, char_to_idx, idx_to_char

# Function to generate input-output sequences from text
def create_sequences(text, char_to_idx, seq_length):
    sequences = []
    next_chars = []
    for i in range(len(text) - seq_length):
        sequences.append([char_to_idx[char] for char in text[i:i + seq_length]])
        next_chars.append(char_to_idx[text[i + seq_length]])
    return sequences, next_chars

# Function to evaluate model performance
def validate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top_k_correct = 0
    k = 5

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            top_k_preds = torch.topk(outputs, k, dim=1).indices
            for i in range(targets.size(0)):
                if targets[i] in top_k_preds[i]:
                    top_k_correct += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    top_k_accuracy = top_k_correct / total
    return avg_loss, accuracy, top_k_accuracy

# Dataset class for text sequences
class TextDataset(Dataset):
    def __init__(self, sequences, next_chars):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.next_chars = torch.tensor(next_chars, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.next_chars[idx]