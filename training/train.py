import argparse
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from training.utils import *  # Import utility functions for text processing
from models.lstm_model import LSTMTextGenerationModel  # Import LSTM model class
from models.transformer_model import TransformerTextGenerationModel  # Import Transformer model class
import json

# Function to save evaluation metrics to a JSON file
def save_metrics(model_type, loss, accuracy, top5_accuracy):
    results = {
        "model": model_type,
        "loss": loss,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy
    }
    with open(f"results/evaluation_metrics.json", "a") as f:
        json.dump(results, f, indent=4)
        f.write("\n")

# Function to train the model
def train_model(model, model_type, train_loader, criterion, optimizer, num_epochs, val_loader=None, test_loader=None, patience=3):
    best_val_loss = float('inf')  # Track best validation loss
    patience_counter = 0  # Counter for early stopping
    model_filename = f"saved_models/best_{model_type}.pth"  # Path to save the best model

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        
        if val_loader:
            val_loss, _, _ = validate_model(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_filename)  # Save the best model
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(torch.load(model_filename))  # Load best model
                break
    
    if test_loader:
        test_loss, test_acc, test_top_k_acc = validate_model(model, test_loader, criterion)
        save_metrics(args.model, test_loss, test_acc, test_top_k_acc)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Top-5 Accuracy: {test_top_k_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], required=True, help="Model type to train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    file_path = "data/pantadeusz.txt"  # Path to training data
    text = load_text(file_path)  # Load text data
    vocab, char_to_idx, idx_to_char = create_vocab(text)  # Create character vocabulary
    sequences, next_chars = create_sequences(text, char_to_idx, seq_length=100)  # Generate training sequences

    # Split data into training, validation, and test sets
    train_seqs, test_seqs, train_targets, test_targets = train_test_split(sequences, next_chars, test_size=0.1, random_state=42)
    train_seqs, val_seqs, train_targets, val_targets = train_test_split(train_seqs, train_targets, test_size=0.1, random_state=42)

    # Create dataset objects
    train_dataset = TextDataset(train_seqs, train_targets)
    val_dataset = TextDataset(val_seqs, val_targets)
    test_dataset = TextDataset(test_seqs, test_targets)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    model = None
    if args.model == "lstm":
        model = LSTMTextGenerationModel(len(vocab), embed_dim=64, hidden_dim=256, num_layers=3).to(device)
    else:
        model = TransformerTextGenerationModel(len(vocab), embed_dim=128, num_heads=8, hidden_dim=1024, num_layers=2).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, args.model, train_loader, criterion, optimizer, num_epochs=100, val_loader=val_loader, test_loader=test_loader)