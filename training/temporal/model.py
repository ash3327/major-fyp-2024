import torch
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

# Define LSTM model
class LSTMGestureModel_Hierachical(nn.Module):
    def __init__(self, body_dim=17, hand_dim=21, num_channels=3, hidden_dim=256, output_dim=128, num_layers=2, dropout=0.2, bidirectional=True, normalize=True):
        super(LSTMGestureModel_Hierachical, self).__init__()
        self.output_dim = output_dim
        self.body_dim = body_dim
        self.hand_dim = hand_dim
        self.normalize = normalize
        self.num_channels = num_channels
        self.lstm_body = nn.LSTM(
            body_dim*num_channels, hidden_dim, num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.lstm_hand = nn.LSTM(
            hand_dim*num_channels, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.lstm_final = nn.LSTM(
            hidden_dim * (2 if bidirectional else 1) * 3, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        # Assume input: [B,L,177]
        # 177 = [17+21+21,3]
        B,L,_ = x.shape
        x = x.view(B,L,self.body_dim+self.hand_dim*2,self.num_channels)
        x = x.clone()
        x_body = x[...,:self.body_dim,:] # body
        x_hand1 = x[...,self.body_dim:self.body_dim+self.hand_dim,:] # hand1
        x_hand2 = x[...,self.body_dim+self.hand_dim:,:] # hand2

        x_hand1[...,:,:3] -= x_hand1[...,0,torch.newaxis,:3]
        x_hand2[...,:,:3] -= x_hand2[...,0,torch.newaxis,:3]

        h_body, _ = self.lstm_body(x_body.view(B,L,-1)) # shape: [B,L,D_in] -> [B,L,D_lstm]
        h_hand1, _ = self.lstm_hand(x_hand1.view(B,L,-1))
        h_hand2, _ = self.lstm_hand(x_hand2.view(B,L,-1))

        h_mid = torch.concatenate([h_body,h_hand1,h_hand2],dim=-1)

        h_final, _ = self.lstm_final(h_mid)      

        out = self.fc(h_final) # shape: -> [B,L,D_out]
        return out
        # return self.fc(x[:, -1, :])  # Take last time step output

# Another LSTM model, but this time with temporal windowing
class LSTMGestureModel_Windowed(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=256, output_dim=128, num_layers=3, dropout=0.2, window_size=16, window_stride=None):
        super(LSTMGestureModel_Windowed, self).__init__()
        self.output_dim = output_dim
        self.window_size = window_size
        if window_stride is None:
            window_stride = window_size
        self.window_stride = window_stride

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # For windowing
        self.temporal_pool = nn.AvgPool1d(kernel_size=window_size, stride=window_stride)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # shape: [B,L,D_in] -> [B,L,D_lstm]
        lstm_out = lstm_out.permute(0,2,1) # [B,D_lstm,L] for windowing
        pooled_out = self.temporal_pool(lstm_out)
        pooled_out = pooled_out.permute(0,2,1) # [B,L//W,D_lstm]
        out = self.fc(pooled_out) # shape: -> [B,L,D_out]
        # if torch.isnan(out).any():
        #     print('\t#####',torch.isnan(x).any(),torch.isnan(pooled_out).any(),torch.isnan(lstm_out).any())
        #     exit(0)
        return out
        # return self.fc(x[:, -1, :])  # Take last time step output

