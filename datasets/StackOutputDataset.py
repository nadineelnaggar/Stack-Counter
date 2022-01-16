import torch
import random
from random import randint

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


# train_inputs, train_states, train_targets, train_actions = createTrainSet()

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


# test_inputs, test_states, test_targets, test_actions = createTestSet()