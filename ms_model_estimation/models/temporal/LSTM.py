import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, inputSize, hiddenState, outputSize, numLayers, dropoutProb, bidirectional=True):
        super(LSTM, self).__init__()

        self.inputSize = inputSize
        self.outputSize = outputSize

        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=hiddenState,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropoutProb,
            bidirectional=bidirectional,
        )
        self.linearLayer = nn.Conv1d(hiddenState if not bidirectional else hiddenState * 2, outputSize, 1, bias=True)
        self.hiddenState = hiddenState if not bidirectional else hiddenState * 2
    def forward(self, x):
        assert x.shape[-1] == self.inputSize

        x, _ = self.lstm(x)


        assert x.shape[-1] == self.hiddenState

        # batch size x hidden state sizes x sequence length
        x = x.permute(0, 2, 1)
        x = self.linearLayer(x)
        # batch size x sequence length x output feature size
        x = x.permute(0, 2, 1)

        return x

    def set_bn_momentum(self, momentum):
        pass
