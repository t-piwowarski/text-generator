import torch
import torch.nn as nn
import math

# Positional Encoding for Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)  # Initialize positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Position indices
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)  # Add positional encoding to embeddings

# Transformer-based Text Generation Model
class TransformerTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, hidden_dim=1024, num_layers=2, max_seq_len=200, dropout=0.2):
        super(TransformerTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)  # Positional encoding layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)  # Transformer decoder layer
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)  # Stacked decoder layers
        self.fc = nn.Linear(embed_dim, vocab_size)  # Final fully connected layer

    def forward(self, x):
        embeddings = self.embedding(x)  # Convert input indices to embeddings
        embeddings = self.positional_encoding(embeddings).transpose(0, 1)  # Apply positional encoding
        tgt_mask = self.generate_square_subsequent_mask(embeddings.size(0)).to(x.device)  # Generate sequence mask
        output = self.transformer_decoder(embeddings, embeddings, tgt_mask=tgt_mask)  # Pass through Transformer decoder
        return self.fc(output.transpose(0, 1)[:, -1, :])  # Generate final output

    @staticmethod
    def generate_square_subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)  # Upper triangular mask for causal attention
        return mask