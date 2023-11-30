import torch
from torch import nn

class CustomEncoder(nn.Module):
    """Custom Encoder module."""
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(CustomEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell

class CustomDecoder(nn.Module):
    """Custom Decoder module."""
    def __init__(self, output_dim, embedding_dim, hidden_dim):
        super(CustomDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell
