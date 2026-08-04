import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super(StockLSTM, self).__init__() 
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True            

        )

        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        model_output, (hidden, cell) = self.lstm(x)
        last_output = model_output[:, -1, :]
        logits = self.linear(last_output)

        return logits