import torch
import torch.nn.functional as F
import argparse
from training.utils import *  # Import utility functions
from models.lstm_model import LSTMTextGenerationModel  # Import LSTM model class
from models.transformer_model import TransformerTextGenerationModel  # Import Transformer model class

# Function to save generated text into a results file
def save_generated_text(model_type, prompt, generated_text):
    with open(f"results/generated_samples/{model_type}_samples.txt", "a") as f:
        f.write(f"Prompt: {prompt}\nGenerated: {generated_text}\n{'-'*50}\n")

# Function to generate text using a trained model
def generate_text(model, start_text, char_to_idx, idx_to_char, seq_length, num_generate, temperature=0.4):
    model.eval()  # Set model to evaluation mode
    input_seq = [char_to_idx[char] for char in start_text]  # Convert input text to indices
    generated_text = start_text  # Initialize generated text with prompt
    
    for _ in range(num_generate):
        input_tensor = torch.tensor(input_seq[-seq_length:], dtype=torch.long).unsqueeze(0).to(model.fc.weight.device)
        
        with torch.no_grad():  # Disable gradient calculations for efficiency
            output = model(input_tensor)  # Get model predictions
            probabilities = F.softmax(output / temperature, dim=-1)  # Apply temperature scaling
            next_char_idx = torch.multinomial(probabilities, 1).item()  # Sample next character
        
        generated_text += idx_to_char[next_char_idx]  # Append new character to generated text
        input_seq.append(next_char_idx)  # Update input sequence
    
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "transformer"], required=True, help="Model type to use for text generation")
    parser.add_argument("--prompt", type=str, required=True, help="Starting text for generation")
    parser.add_argument("--num_chars", type=int, default=400, help="Number of characters to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    file_path = "data/pantadeusz.txt"  # Path to training data
    text = load_text(file_path)  # Load text data
    vocab, char_to_idx, idx_to_char = create_vocab(text)  # Create character vocabulary

    model = None
    if args.model == "lstm":
        model = LSTMTextGenerationModel(len(vocab), embed_dim=64, hidden_dim=256, num_layers=3).to(device)
        model.load_state_dict(torch.load("saved_models/best_lstm.pth"))  # Load trained LSTM model
    else:
        model = TransformerTextGenerationModel(len(vocab), embed_dim=128, num_heads=8, hidden_dim=1024, num_layers=2).to(device)
        model.load_state_dict(torch.load("saved_models/best_transformer.pth"))  # Save generated text to file
    
    generated_text = generate_text(model, args.prompt, char_to_idx, idx_to_char, seq_length=100, num_generate=args.num_chars)
    save_generated_text(args.model, args.prompt, generated_text)
    print("Generated text:")
    print(generated_text)