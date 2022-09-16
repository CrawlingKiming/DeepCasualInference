import torch.nn as nn
import torch
from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module


class MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu'):
        layers = []
        #if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
            layers.append(nn.Dropout(p=0.15))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        #if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(MLP, self).__init__(*layers)


class RNN(nn.Module):
    def __init__(self,hidden_units, n_class=1, n_hidden=1):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.1, batch_first= True)
        self.mlp = MLP(16+20, 1, hidden_units=hidden_units)
        #self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        #self.b = nn.Parameter(torch.randn([n_class]).type(dtype))
        #self.Softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # X : Z(1) X(5) V(10)
        Xseq = X[:, 1:11]#.view(-1,-1,1)
        Xseq = Xseq.unsqueeze(2)
        Z = X[:, [0]]
        V = X[:, 11:]
        #X = X.transpose(0, 1)
        outputs, hidden = self.rnn(Xseq)
        outputs = outputs[:, :, 0]  # 최종 예측 Hidden Layer
        final_output = torch.cat((Z, outputs, V), dim=1)
        result = self.mlp(final_output)  # 최종 예측 최종 출력 층
        return result

class SWAN_LSTM(nn.Module):
    def __init__(self, hidden_units, n_class=22, n_hidden=15):
        super(SWAN_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, batch_first=True)
        self.mlp = MLP(n_hidden, 4, hidden_units=hidden_units)
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        """
        X : (N,L,H) # Batch X Sequence Length X Hidden num

        :param X:
        :return: N * L * 4
        """
        #print(X.shape)
        outputs, hidden = self.lstm(X)

        N, L, H = outputs.shape # Batch * LEngth ^ hiddden_size
        #print(outputs.shape, H, N*L)
        #output.shape
        outputs = outputs.reshape(-1, H)# N X L X H #[:, -1, :]  # Use Only Last Sequence

        out = self.mlp(outputs)  # 최종 예측 최종 출력 층
        result = self.softmax(out)
        result = result.reshape(N, L, 4)
        #print(result)
        #raise ValueError
        return result

class RNN_full(nn.Module):
    def __init__(self,hidden_units, n_class=1, n_hidden=1):
        super(RNN_full, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.2, batch_first= True)
        self.mlp = MLP(16+20, 1, hidden_units=hidden_units)
        #self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))
        #self.b = nn.Parameter(torch.randn([n_class]).type(dtype))
        #self.Softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # X : Z(1) X(5) V(10)
        Xseq = X[:, 1:]#.view(-1,-1,1)
        Xseq = Xseq.unsqueeze(2)
        Z = X[:, [0]]
        #V = X[:, 6:]
        #X = X.transpose(0, 1)
        outputs, hidden = self.rnn(Xseq)
        outputs = outputs[:, :, 0]  # 최종 예측 Hidden Layer
        final_output = torch.cat((Z, outputs), dim=1)
        result = self.mlp(final_output)  # 최종 예측 최종 출력 층
        return result

def model_fetcher(args):
    if args.model == "MLP" : return MLP(16+20, 1, args.hidden_units)
    elif args.model == "RNN" : return RNN(hidden_units=args.hidden_units)
    elif args.model == "RNN_FULL" : return RNN_full(hidden_units=args.hidden_units)
    elif args.model == "SWAN_LSTM":
        return SWAN_LSTM(hidden_units=args.hidden_units)
