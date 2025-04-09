import torch.nn as nn

# Define LSTM model
class LSTMGestureModel(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, output_dim=128, num_layers=3, dropout=0.2):
        super(LSTMGestureModel, self).__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x) # shape: [B,L,D_in] -> [B,L,D_lstm]
        out = self.fc(x) # shape: -> [B,L,D_out]
        return out
        # return self.fc(x[:, -1, :])  # Take last time step output
