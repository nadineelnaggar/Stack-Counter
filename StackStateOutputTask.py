import torch
import torch.nn as nn
import torch.optim as optim
import Stack_Counter
from Stack_Counter import StackCounterNN
import random
from random import random, randint
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix


device = torch.cuda if torch.cuda.is_available() else 'cpu'
input_size = 2
hidden_size = 2


class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.stack = StackCounterNN()
        self.out = nn.Softmax(dim=0)
        # self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.stack(x)
        # x = self.out(x)
        return x

    def reset(self):
        self.stack.reset()


model = Net(input_size, hidden_size)


# threshold_push = 0.5
# threshold_pop = 0.5

def generate_push():
    inp = torch.tensor([1, 0], dtype=torch.float32, requires_grad=True)
    # print(inp)
    rand_stackDepth = randint(1, 5)
    rand_falsepop = randint(0, 5)
    target = torch.tensor([rand_stackDepth + 1, rand_falsepop], dtype=torch.float32)
    state = [rand_stackDepth,rand_falsepop]
    action = 'Push'
    return inp, state, target, action




def generate_pop():
    inp = torch.tensor([0, 1], dtype=torch.float32, requires_grad=True)
    rand_stackDepth = randint(1, 5)
    rand_falsepop = randint(0, 5)
    if rand_stackDepth > 0:
        target = torch.tensor([rand_stackDepth - 1, rand_falsepop], dtype=torch.float32)
    elif rand_stackDepth == 0:
        target = torch.tensor([rand_stackDepth, rand_falsepop + 1], dtype=torch.float32)
    state = [rand_stackDepth, rand_falsepop]
    action = 'Pop'
    return inp, state, target, action


def generate_noop():
    inp = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True)
    rand_stackDepth = randint(1, 5)
    rand_falsepop = randint(0, 5)
    target = torch.tensor([rand_stackDepth, rand_falsepop], dtype=torch.float32)
    state = [rand_stackDepth, rand_falsepop]
    action = 'NoOp'
    return inp, state, target, action


def generate_push_empty():
    inp = torch.tensor([1, 0], dtype=torch.float32, requires_grad=True)
    # print(inp)
    rand_stackDepth = 0
    rand_falsepop = randint(0, 5)
    target = torch.tensor([rand_stackDepth + 1, rand_falsepop], dtype=torch.float32)
    state = [rand_stackDepth, rand_falsepop]
    action = 'Push Empty'
    return inp, state, target, action




def generate_pop_empty():
    inp = torch.tensor([0, 1], dtype=torch.float32, requires_grad=True)
    rand_stackDepth = 0
    rand_falsepop = randint(0, 5)
    # model.stack.editStackState(rand_stackDepth,rand_falsepop)
    if rand_stackDepth > 0:
        target = torch.tensor([rand_stackDepth - 1, rand_falsepop], dtype=torch.float32)
    elif rand_stackDepth == 0:
        target = torch.tensor([rand_stackDepth, rand_falsepop + 1], dtype=torch.float32)
    state = [rand_stackDepth, rand_falsepop]
    action = 'Pop Empty'
    return inp, state, target, action


def generate_noop_empty():
    inp = torch.tensor([0, 0], dtype=torch.float32, requires_grad=True)
    rand_stackDepth = 0
    rand_falsepop = randint(0, 5)
    target = torch.tensor([rand_stackDepth, rand_falsepop], dtype=torch.float32)
    state = [rand_stackDepth, rand_falsepop]
    action = 'NoOp Empty'
    return inp, state, target, action

def createTrainSet():
    count_push = 0
    count_pop = 0
    count_noop = 0
    count_push_empty = 0
    count_pop_empty = 0
    count_noop_empty = 0
    input_data = []
    state_data = []
    target_data = []
    action_data=[]
    for i in range(6000):
        state = randint(0,5)
        # state = randint(0, 1)
        # state = 0 --> Push (non-empty stack)
        # state = 1 --> Pop (non-empty stack)
        # state = 2 --> NoOp (non-empty stack)
        # state = 3 --> Push (empty stack)
        # state = 4 --> Pop (empty stack (false pop))
        # state = 5 --> NoOp (empty stack)

        if state == 0:

            inp, state, target, action = generate_push()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_push += 1
        elif state == 1:

            inp, state, target, action = generate_pop()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_pop += 1
        elif state == 2:

            inp, state, target, action = generate_noop()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_noop += 1
        elif state == 3:

            inp, state, target, action = generate_push_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_push_empty += 1
        elif state == 4:

            inp, state, target, action = generate_pop_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_pop_empty += 1
        elif state == 5:

            inp, state, target, action = generate_noop_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_noop_empty += 1
    print('count_push = ', count_push)
    print('count_pop = ', count_pop)
    print('count_noop = ', count_noop)
    print('count_push_empty = ',count_push_empty)
    print('count_pop_empty = ',count_pop_empty)
    print('count_noop_empty = ',count_noop_empty)
    print('len(input_data) = ', len(input_data))
    return input_data, state_data, target_data, action_data


train_inputs, train_states, train_targets, train_actions = createTrainSet()
print(train_inputs[10])
print(train_states[10])
print(train_targets[10])
print(train_actions[10])

print_steps = 5000
plot_steps = 1000

learning_rate=0.0003
# learning_rate=0.001
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor):

    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    return output, loss


n_correct = 0
n_iters = 100000
current_loss = 0
current_correct = 0
expected_actions = []
actual_actions = []
# actual_action = None

n_epochs = 5

for epoch in range(n_epochs):
    current_correct=0
    for idx in range(len(train_inputs)):
        input_tensor = train_inputs[idx]
        state_tensor = train_states[idx]
        target_tensor = train_targets[idx]
        expected_action = train_actions[idx]
        expected_actions.append(expected_action)
        model.stack.editStackState(state_tensor[0], state_tensor[1])
        output, loss = train(input_tensor, target_tensor)
        if output[0] == target_tensor[0] and output[1] == target_tensor[1]:
            # n_correct += 1
            current_correct += 1
        if epoch==n_epochs-1:
            if output[0] == state_tensor[0] and output[1] == state_tensor[1] and state_tensor[0] > 0:
                actual_action = 'NoOp'
                actual_actions.append(actual_action)
            elif output[0] == state_tensor[0] and output[1] == state_tensor[1] and state_tensor[0] == 0:
                actual_action = 'NoOp Empty'
                actual_actions.append(actual_action)
            elif output[0] == (state_tensor[0] + 1) and output[1] == state_tensor[1] and state_tensor[0] > 0:
                actual_action = 'Push'
                actual_actions.append(actual_action)
            elif output[0] == (state_tensor[0] + 1) and output[1] == state_tensor[1] and state_tensor[0] == 0:
                actual_action = 'Push Empty'
                actual_actions.append(actual_action)
            elif output[0] == (state_tensor[0] - 1) and output[1] == state_tensor[1] and state_tensor[1] >= 0:
                actual_action = 'Pop'
                actual_actions.append(actual_action)
            elif output[0] == state_tensor[0] and state_tensor[0] == 0 and output[1] == (state_tensor[1] + 1):
                actual_action = 'Pop Empty'
                actual_actions.append(actual_action)

    print('Accuracy for epoch ',epoch, ' = ',current_correct/len(train_inputs)*100)
    if epoch == n_epochs - 1:
        print('model weights = ', model.fc1.weight)
        print('model bias = ', model.fc1.bias)
        print('Training Accuracy = ',current_correct/len(train_inputs)*100)
        conf_matrix = sklearn.metrics.confusion_matrix(train_actions, actual_actions)
        print(conf_matrix)
        operations = ['Push', 'Pop', 'NoOp', 'Push Empty', 'Pop Empty', 'NoOp Empty']

        heat = sns.heatmap(conf_matrix, xticklabels=operations, yticklabels=operations, annot=True, fmt="d")
        bottom1, top1 = heat.get_ylim()
        heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        plt.show()






"""
Inputs needed
    - Push to empty stack
    - Push to non empty stack
    - Pop from empty stack (false pop)
    - Pop from non empty stack
    - NoOp on empty stack
    - NoOp on non empty stack


Outputs:
    - Stack Depth (increment, decrement, unchanged)
    - False Pop Count (increment, unchanged)

    - Stack Depth increment, false pop increment
    - Stack Depth increment, false pop unchanged
    - Stack depth decrement, false pop increment
    - Stack depth decrement, false pop unchanged
    - Stack depth unchanged, false pop increment
    - Stack depth unchanged, false pop unchanged

"""


def createTestSet():
    count_push = 0
    count_pop = 0
    count_noop = 0
    count_push_empty = 0
    count_pop_empty = 0
    count_noop_empty = 0
    input_data = []
    state_data = []
    target_data = []
    action_data=[]
    for i in range(2000):
        state = randint(0,5)
        # state = randint(0, 1)
        # state = 0 --> Push (non-empty stack)
        # state = 1 --> Pop (non-empty stack)
        # state = 2 --> NoOp (non-empty stack)
        # state = 3 --> Push (empty stack)
        # state = 4 --> Pop (empty stack (false pop))
        # state = 5 --> NoOp (empty stack)

        if state == 0:

            inp, state, target, action = generate_push()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_push += 1
        elif state == 1:

            inp, state, target, action = generate_pop()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_pop += 1
        elif state == 2:

            inp, state, target, action = generate_noop()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_noop += 1
        elif state == 3:

            inp, state, target, action = generate_push_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_push_empty += 1
        elif state == 4:

            inp, state, target, action = generate_pop_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_pop_empty += 1
        elif state == 5:

            inp, state, target, action = generate_noop_empty()
            input_data.append(inp)
            state_data.append(state)
            target_data.append(target)
            action_data.append(action)
            count_noop_empty += 1
    print('count_push = ', count_push)
    print('count_pop = ', count_pop)
    print('count_noop = ', count_noop)
    print('count_push_empty = ',count_push_empty)
    print('count_pop_empty = ',count_pop_empty)
    print('count_noop_empty = ',count_noop_empty)
    print('len(input_data) = ', len(input_data))
    return input_data, state_data, target_data, action_data

model.eval()

test_inputs, test_states, test_targets, test_actions = createTestSet()

test_expected_actions = []
test_actual_actions = []
n_correct_test = 0

with torch.no_grad():
    print('model weights = ',model.fc1.weight)
    print('model bias = ',model.fc1.bias)

    for i in range(2000):

        input_tensor = test_inputs[i]
        state_tensor = test_states[i]
        target_tensor = test_targets[i]
        expected_action = test_actions[i]
        test_expected_actions.append(expected_action)
        model.stack.editStackState(state_tensor[0],state_tensor[1])
        output = model(input_tensor)

        if output[0] == target_tensor[0] and output[1] == target_tensor[1]:
            n_correct_test += 1

        if output[0]==state_tensor[0] and output[1]==state_tensor[1] and state_tensor[0]>0:
            actual_action = 'NoOp'
            test_actual_actions.append(actual_action)
        elif output[0]==state_tensor[0] and output[1]==state_tensor[1] and state_tensor[0]==0:
            actual_action='NoOp Empty'
            test_actual_actions.append(actual_action)
        elif output[0]==(state_tensor[0]+1) and output[1]==state_tensor[1] and state_tensor[0]>0:
            actual_action='Push'
            test_actual_actions.append(actual_action)
        elif output[0]==(state_tensor[0]+1) and output[1]==state_tensor[1] and state_tensor[0]==0:
            actual_action='Push Empty'
            test_actual_actions.append(actual_action)
        elif output[0]==(state_tensor[0]-1) and output[1]==state_tensor[1] and state_tensor[1]>=0:
            actual_action='Pop'
            test_actual_actions.append(actual_action)
        elif output[0]==state_tensor[0] and state_tensor[0]==0 and output[1]==(state_tensor[1]+1):
            actual_action='Pop Empty'
            test_actual_actions.append(actual_action)


print('Test Accuracy = ', (n_correct_test / 2000) * 100, '%')

conf_matrix1 = sklearn.metrics.confusion_matrix(test_expected_actions,test_actual_actions)
print(conf_matrix1)
heat1 = sns.heatmap(conf_matrix1,xticklabels=operations,yticklabels=operations,annot=True,fmt="d",cmap="YlGnBu")
bottom2, top2 = heat1.get_ylim()
heat1.set_ylim(bottom2 + 0.5, top2 - 0.5)
plt.show()


