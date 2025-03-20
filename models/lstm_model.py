import torch.nn as nn

class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=3, dropout=0.2):
        super(LSTMTextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Converts input characters to dense vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)  # LSTM layers
        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Fully connected output layer

    def forward(self, x):
        x = self.embedding(x)  # Pass input through embedding layer
        lstm_out, _ = self.lstm(x)  # Pass embeddings through LSTM layers
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Apply dropout to the last LSTM output
        return self.fc(lstm_out)  # Pass through final fully connected layer