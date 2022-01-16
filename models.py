import torch
import torch.nn as nn
import Stack_Counter
from Stack_Counter import StackCounterNN



class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(VanillaRNN, self).__init__()
        self.model_name='VanillaRNN'
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.task=task
        self.initialisation = initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        if freeze_input_layer==True:
            self.fc1.weight = nn.Parameter(torch.eye(hidden_size), requires_grad=False)
            self.fc1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.rnn = nn.RNN(hidden_size,hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size,output_size)
        if freeze_output_layer==True:
            self.fc2.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32), requires_grad=False)
            self.fc2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.fc1(x)
        x, hidden = self.rnn(x.unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        x = x.squeeze()
        x = self.fc2(x)
        if self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation=='Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(VanillaLSTM, self).__init__()
        self.model_name='VanillaLSTM'
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.task=task
        self.initialisation = initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        if freeze_input_layer==True:
            self.fc1.weight = nn.Parameter(torch.eye(hidden_size), requires_grad=False)
            self.fc1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.lstm=nn.LSTM(hidden_size,hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size,output_size)
        if freeze_output_layer==True:
            self.fc2.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32), requires_grad=False)
            self.fc2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.fc1(x)
        x, hidden = self.lstm(x.unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        x = x.squeeze()
        x = self.fc2(x)
        if self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation=='Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden

class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random',task='BinaryClassification'):
        super(VanillaGRU, self).__init__()
        self.model_name='VanillaGRU'
        self.task=task
        self.initialisation = initialisation
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        if freeze_input_layer==True:
            self.fc1.weight = nn.Parameter(torch.eye(hidden_size), requires_grad=False)
            self.fc1.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.gru = nn.GRU(hidden_size,hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if freeze_output_layer==True:
            self.fc2.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32), requires_grad=False)
            self.fc2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,hidden):
        x = self.fc1(x)
        x, hidden = self.gru(x.unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        x = x.squeeze()
        x = self.fc2(x)
        if self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
        elif self.output_activation=='Clipping':
            x = torch.clamp(x, min=0, max=1)
        return x, hidden


class RecurrentStack(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification', bias=False):
        super(RecurrentStack, self).__init__()
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.model_name='RecurrentStack'
        self.task=task
        self.initialisation = initialisation
        self.freeze_input_layer=freeze_input_layer
        self.freeze_output_layer=freeze_output_layer
        self.bias=bias
        # self.fc1=nn.Linear(input_size,stack_input_size, bias=False)
        # self.fc1 = nn.Linear(input_size,stack_input_size,bias=self.bias)
        if self.bias==True:
            self.fc1 = nn.Linear(input_size,stack_input_size)
        elif self.bias==False:
            self.fc1 = nn.Linear(input_size,stack_input_size, bias=False)
        if initialisation=='Correct':
            self.fc1.weight=nn.Parameter(torch.eye(stack_input_size))
            if self.bias==True:
                self.fc1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        if freeze_input_layer==True:
            self.fc1.weight=nn.Parameter(torch.eye(stack_input_size), requires_grad=False)
            if self.bias==True:
                self.fc1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32), requires_grad=False)
        self.stack=StackCounterNN()
        # self.fc2 = nn.Linear(stack_output_size,output_size, bias=False)
        # self.fc2=nn.Linear(stack_output_size,output_size, bias=self.bias)
        if self.bias==True:
            self.fc2 = nn.Linear(stack_output_size,output_size)
        elif self.bias==False:
            self.fc2 = nn.Linear(stack_output_size, output_size, bias=False)
        if initialisation=='Correct':
            self.fc2.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32))
            if self.bias==True:
                self.fc2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32))
        if freeze_output_layer==True:
            self.fc2.weight=nn.Parameter(torch.tensor([1,1],dtype=torch.float32), requires_grad=False)
            if self.bias==True:
                self.fc2.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32), requires_grad=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,stack_state):
        x = self.fc1(x)
        x = self.stack(x, stack_state)
        stack_state=x

        if self.output_activation=='Sigmoid':
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = x.squeeze()
        elif self.output_activation=='Clipping':
            x = self.fc2(x)
            x = torch.clamp(x, min=0, max=1)
            # x = x.unsqueeze(dim=0)
            x = x.squeeze()
        elif self.output_activation=='None':
            return x, stack_state
        return x, stack_state






class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(StackLSTM, self).__init__()
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.model_name='StackLSTM'
        self.task=task
        self.initialisation = initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size,hidden_size+stack_input_size)
        # if freeze_input_layer==True:
        #     self.fc1.weight = nn.Parameter(torch.eye(2),requires_grad=False)
        #     self.fc1.bias = nn.Parameter(torch.tensor(0,dtype=torch.float32),requires_grad=False)
        self.stack = StackCounterNN()
        self.lstm = nn.LSTM(hidden_size,hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(stack_output_size+hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        lstm_output, hidden= self.lstm(x[:self.hidden_size].unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        # x = x.squeeze()
        # x = self.stack(x, stack_state)
        # stack_state=x
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, lstm_output.squeeze()))
        # combined = torch.cat((x, lstm_output.squeeze()))
        x = self.fc2(combined)
        if self.output_activation=='Sigmoid':
            x = self.sigmoid(x)
            x = x.squeeze()
        elif self.output_activation=='Clipping':
            x = torch.clamp(x, min=0, max=1)
            # x = x.unsqueeze(dim=0)
            x=x.squeeze()
        return x, hidden, stack_state


class StackRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(StackRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.model_name = 'StackRNN'
        self.task = task
        self.initialisation = initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size, hidden_size+stack_input_size)
        self.stack = StackCounterNN()
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(stack_output_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        rnn_output, hidden = self.rnn(x[:self.hidden_size].unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        # x = x.squeeze()
        # x = self.stack(x, stack_state)
        # stack_state = x
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        combined = torch.cat((stack_state, rnn_output.squeeze()))
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x)
            x = x.squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1)
            # x = x.unsqueeze(dim=0)
            x=x.squeeze()
        return x, hidden, stack_state

class StackGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(StackGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_activation = output_activation
        self.num_layers=num_layers
        self.model_name = 'StackGRU'
        self.task = task
        self.initialisation = initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1 = nn.Linear(input_size, hidden_size+stack_input_size)
        self.stack = StackCounterNN()
        self.gru= nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(stack_output_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden, stack_state):
        x = self.fc1(x)
        gru_output, hidden = self.gru(x[:self.hidden_size].unsqueeze(dim=0).unsqueeze(dim=0), hidden)
        # x = x.squeeze()
        # x = self.stack(x[:self.hidden_size], stack_state)
        stack_state = self.stack(x[self.hidden_size:].squeeze(), stack_state)
        # stack_state = x
        # combined = torch.cat((x, gru_output.squeeze()))
        combined=torch.cat((stack_state,gru_output.squeeze()))
        x = self.fc2(combined)
        if self.output_activation == 'Sigmoid':
            x = self.sigmoid(x)
            x = x.squeeze()
        elif self.output_activation == 'Clipping':
            x = torch.clamp(x, min=0, max=1)
            # x = x.unsqueeze(dim=0)
            x = x.squeeze()
        return x, hidden, stack_state



class FFStack(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, stack_input_size=2, stack_output_size=2,
                 freeze_input_layer=False, freeze_output_layer=False, output_activation='Sigmoid',
                 initialisation='Random', task='BinaryClassification'):
        super(FFStack, self).__init__()
        self.hidden_size=hidden_size
        self.output_activation=output_activation
        self.model_name='FFStack'
        self.task=task
        self.initialisation=initialisation
        self.freeze_input_layer = freeze_input_layer
        self.freeze_output_layer = freeze_output_layer
        self.fc1=nn.Linear(input_size,stack_input_size, bias=False)
        if freeze_input_layer==True:
            self.fc1.weight=nn.Parameter(torch.eye(stack_input_size),requires_grad=False)
        elif initialisation=='Correct' and freeze_input_layer==False:
            self.fc1.weight=nn.Parameter(torch.eye(stack_input_size))
        self.stack=StackCounterNN()
        self.fc2=nn.Linear(stack_output_size, output_size, bias=False)
        if freeze_output_layer==True:
            if output_activation=='Sigmoid':
                self.fc2.weight=nn.Parameter(torch.tensor([40,40],dtype=torch.float32),requires_grad=False)
            elif output_activation=='Clipping':
                self.fc2.weight = nn.Parameter(torch.tensor([1, 1], dtype=torch.float32), requires_grad=False)
        elif initialisation=='Correct' and freeze_output_layer==False:
            if output_activation=='Sigmoid':
                self.fc2.weight = nn.Parameter(torch.tensor([40, 40], dtype=torch.float32))
            elif output_activation=='Clipping':
                self.fc2.weight = nn.Parameter(torch.tensor([1, 1], dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.stack(x)
        if self.task=='StackStateOutput':
            return x

        if self.output_activation == 'Sigmoid':
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = x.squeeze()
        elif self.output_activation == 'Clipping':
            x = self.fc2(x)
            x = torch.clamp(x, min=0, max=1)
            # x = x.unsqueeze(dim=0)
            x = x.squeeze()
        elif self.output_activation=='None':
            return x
        return x


