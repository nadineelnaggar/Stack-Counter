import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
import Stack_Counter
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models import VanillaGRU, VanillaLSTM, VanillaRNN, FFStack, RecurrentStack, StackGRU, StackLSTM, StackRNN



##############################
# CONFIGURE THE MODEL AND EXPERIMENTS

# Stack models
model_name = 'RecurrentStack'
# model_name = 'FFStack'

# Mixed models
# model_name = 'StackLSTM'
# model_name = 'StackRNN'
# model_name = 'StackGRU'

# Vanilla models
# model_name = 'VanillaGRU'
# model_name = 'VanillaLSTM'
# model_name = 'VanillaRNN'


# num_bracket_pairs = 1
num_bracket_pairs=2
length_bracket_pairs=6

input_size=2

# hidden_size=1
hidden_size=2
# hidden_size=3
# hidden_size=4

num_layers = 1
# num_layers=2

stack_input_size=2
stack_output_size=2
output_size = 1

output_activation='Clipping'
# output_activation='Sigmoid'

freeze_input_layer=False
# freeze_input_layer=True

freeze_output_layer=False
# freeze_output_layer=True

# initialisation='Random'
initialisation='Correct'

use_optimiser='Adam'
# use_optimiser='SGD'

learning_rate=0.001
num_runs=10
# num_epochs=1000
num_epochs=2000
# num_epochs=10000

# bias=False
bias=True

task = 'BinaryClassification'
# task='StackStateOutput'

if model_name!='FFStack' and model_name!='RecurrentStack':
    task='BinaryClassification'




###############################################################################

frozen_layers = 'no_frozen_layers'
if freeze_input_layer==True and freeze_output_layer==False:
    frozen_layers='frozen_input_layer'
elif freeze_input_layer==False and freeze_output_layer==True:
    frozen_layers='frozen_output_layer'





file_name = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'.txt'
excel_name = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'.xlsx'
modelname = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_MODEL.pth'
optimname = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_OPTIMISER.pth'
train_log= 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_TRAIN_LOG.txt'
test_log = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_TEST_LOG.txt'
long_test_log = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_LONG_TEST_LOG.txt'
plot_name = 'ERROR ANALYSIS Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'_PLOT.png'





with open(train_log,'w') as f:
    f.write('\n')
with open(test_log,'w') as f:
    f.write('\n')
with open(file_name,'w') as f:
    f.write('\n')
with open(long_test_log,'w') as f:
    f.write('\n')



vocab=['(',')']

num_classes = 2
data = []
X = []
stack_states = []
y = []
n_letters = len(vocab)
input_size = n_letters
labels = ['valid','invalid']


def classFromOutput(output):
    # print('output.item() = ',output.item())
    # category_i = int(output.item())
    if output.item()>0.5:
        category_i=1
    else:
        category_i=0
    # print('category_i=',category_i)
    return labels[category_i],category_i




"""

4 cases to evaluate in the case of FFStack model on the BinaryClassification task
    1. ( on the stack, ) input to the network. Valid sequence.
    2. ( on the stack, ( input to the network. Invalid sequence.
    3. ) on the stack, ( input to the network. Invalid sequence.
    4. ) on the stack, ) input to the network. Invalid sequence.

    Oversample valid class.

"""

if model_name=='FFStack' and task=='BinaryClassification':
    X.append(')')
    y.append('valid')
    stack_states.append([1, 0])

    X.append('(')
    y.append('invalid')
    stack_states.append([1, 0])

    X.append(')')
    y.append('valid')
    stack_states.append([1, 0])

    X.append('(')
    y.append('invalid')
    stack_states.append([0, 1])

    X.append(')')
    y.append('valid')
    stack_states.append([1, 0])

    X.append(')')
    y.append('invalid')
    stack_states.append([0, 1])

    max_length = 1

    print('X dataset = ', X)
    print('y dataset = ', y)
    print('stack states = ', stack_states)

# elif model_name=='RecurrentStack' and task=='BinaryClassification':
elif model_name!='FFStack' and task=='BinaryClassification':
    if num_bracket_pairs == 1:
        with open("datasets/Dyck1_Dataset_1pair_balanced.txt", 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                X.append(sentence)
                y.append(label)
                data.append((sentence, label))

        max_length = 2
    elif num_bracket_pairs == 2:
        with open("datasets/Dyck1_Dataset_2pairs_balanced.txt", 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                X.append(sentence)
                y.append(label)
                data.append((sentence, label))

        max_length = 4

    elif num_bracket_pairs == 6:
        with open("datasets/Dyck1_Dataset_6pairs_balanced.txt", 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                X.append(sentence)
                y.append(label)
                data.append((sentence, label))

        max_length = 12

def encode_sentence(sentence, dataset='short'):
    max_length=1
    if dataset=='short' and model_name!='FFStack' and task=='BinaryClassification':
        max_length=2*num_bracket_pairs
    elif dataset=='long':
        max_length=2*length_bracket_pairs
    rep = torch.zeros(max_length,1,n_letters)
    if len(sentence)<max_length:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index+(max_length-len(sentence))][0][pos] = 1
    else:
        for index, char in enumerate(sentence):
            pos = vocab.index(char)
            rep[index][0][pos]=1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):

    if output_activation=='Sigmoid' or output_activation=='Clipping':
        # return torch.tensor([labels.index(label)], dtype=torch.float32)
        if model_name=='VanillaRNN' or model_name=='VanillaLSTM' or model_name=='VanillaGRU':
            return torch.tensor([labels.index(label)],dtype=torch.float32)
        else:
            return torch.tensor(labels.index(label), dtype=torch.float32)
    elif output_activation=='Softmax':
        if label == 'valid':
            return torch.tensor([1,0],dtype=torch.float32)
        elif label == 'invalid':
            return torch.tensor([0,1],dtype=torch.float32)


def encode_dataset(sentences,labels, dataset='short'):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence, dataset))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


if model_name=='FFStack' and task=='BinaryClassification':
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=False)
    stack_states_train = stack_states[:4]
    print(stack_states_train)
    stack_states_test = stack_states[4:]
    print(stack_states_test)
elif model_name!='FFStack' and task=='BinaryClassification':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)




print('length of training set = ',len(X_train))
print('length of test set = ',len(X_test))

X_notencoded = X
y_notencoded = y
X_train_notencoded = X_train
y_train_notencoded = y_train
X_test_notencoded = X_test
y_test_notencoded = y_test
X_train, y_train = encode_dataset(X_train,y_train)
X_test, y_test = encode_dataset(X_test,y_test)
X_encoded, y_encoded = encode_dataset(X,y)

X_long = []
y_long = []
data_long = []


if model_name!='FFStack' and task=='BinaryClassification':
    with open("datasets/Dyck1_Dataset_6pairs_balanced.txt", 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            X_long.append(sentence)
            y_long.append(label)
            data.append((sentence, label))

    X_long_notencoded = X_long
    y_long_notencoded=y_long
    X_long, y_long = encode_dataset(X_long, y_long, dataset='long')



if model_name=='FFStack' and task=='BinaryClassification':
    stack_states_train = stack_states[:4]
    print(stack_states_train)
    stack_states_test = stack_states[4:]
    print(stack_states_test)


train_accuracies = []
train_dataframes = []
test_accuracies = []
long_test_accuracies = []


def train_model(model, task='BinaryClassification'):

    criterion = nn.MSELoss()
    if use_optimiser=='Adam':
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif use_optimiser=='SGD':
        optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    epochs = []
    df1 = pandas.DataFrame()

    initial_weights_input_layer = []
    initial_biases_output_layer=[]
    initial_weights_output_layer = []
    initial_biases_output_layer = []
    initial_gradients_input_layer = []
    initial_gradients_output_layer = []

    input_token=[]
    initial_stack_depth = []
    initial_false_pop_count=[]
    final_stack_depth=[]
    output_label=[]

    weights_input_layer = []
    biases_input_layer = []
    weights_output_layer = []
    biases_output_layer = []
    gradients_input_layer = []
    gradients_output_layer = []
    confusion_matrices = []
    all_losses = []
    current_loss = 0
    all_epoch_incorrect_guesses = []
    accuracies = []
    print_flag = False

    for epoch in range(num_epochs):
        epochs.append(epoch)

        if epoch==(num_epochs-1):
            print_flag=True
        if print_flag==True:
            with open(train_log, 'a') as f:
                f.write('\nEPOCH ' + str(epoch) + '\n')
        confusion = torch.zeros(num_classes,num_classes)
        num_correct = 0
        num_samples = len(X_train)
        current_loss = 0
        epoch_incorrect_guesses = []
        predicted_classes = []
        expected_classes = []

        for i in range(len(X_train)):
            input_tensor = X_train[i]
            class_tensor = y_train[i]
            input_sentence = X_train_notencoded[i]
            class_category = y_train_notencoded[i]
            if model.model_name=='StackRNN' or model.model_name=='StackLSTM' or model.model_name=='StackGRU' or model.model_name=='RecurrentStack' or model.model_name=='FFStack':
                model.stack.reset()

            if model.model_name =='FFStack':
                model.stack.editStackState(stack_states_train[i][0], stack_states_train[i][1])
            optimiser.zero_grad()

            if print_flag == True:
                with open(train_log,'a') as f:
                    f.write('////////////////////////////////////////\n')
                    f.write('input sentence = ' + input_sentence + '\n')
                    if model.model_name!='VanillaRNN' and model.model_name!='VanillaLSTM' and model.model_name!='VanillaGRU':
                        f.write('initial stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                            str(model.stack.false_pop_count.item()) + ']\n')

                print('////////////////////////////////////////')
                print('input sentence = ', input_sentence)
                if model.model_name != 'VanillaRNN' and model.model_name != 'VanillaLSTM' and model.model_name != 'VanillaGRU':
                    print('initial stack state = [', model.stack.stack_depth.item(), ',',
                      model.stack.false_pop_count.item(), ']')

            if model.model_name=='VanillaLSTM' or model.model_name=='StackLSTM':
                hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
            elif model.model_name=='VanillaRNN' or model.model_name=='VanillaGRU' or model.model_name=='StackRNN' \
                        or model.model_name=='StackGRU':
                hidden = torch.zeros(1,1,hidden_size)
            if model.model_name=='StackLSTM' or model.model_name=='StackGRU' or model.model_name=='StackRNN' or \
                        model.model_name=='RecurrentStack':
                stack_state = torch.tensor([0,0], dtype=torch.float32)


            for j in range(input_tensor.size()[0]):
                if model.model_name=='FFStack':
                    output = model(input_tensor[j].squeeze())
                    if print_flag == True:
                        with open(train_log, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')
                elif model.model_name == 'VanillaLSTM' or model.model_name=='VanillaRNN' or \
                    model.model_name=='VanillaGRU':
                    output, hidden = model(input_tensor[j].squeeze(), hidden)
                elif model.model_name=='StackLSTM' or model.model_name=='StackRNN' or model.model_name=='StackGRU':
                    output, hidden, stack_state = model(input_tensor[j].squeeze(), hidden, stack_state)
                    if print_flag == True:
                        with open(train_log, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')
                elif model.model_name=='RecurrentStack':
                    output, stack_state = model(input_tensor[j].squeeze(), stack_state)
                    if print_flag == True:
                        with open(train_log, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')
            loss = criterion(output, class_tensor)
            if print_flag == True:
                print('Loss = ', loss)

            loss.backward()
            optimiser.step()
            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('output in train function = ' + str(output) + '\n')
                    print('//////////////////////////////////////////\n')
                print('output in train function = ', output)

            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            confusion[class_i][guess_i] += 1
            current_loss += loss
            expected_classes.append(class_i)
            predicted_classes.append(guess_i)
            if guess == class_category:
                num_correct += 1
            else:
                if model.model_name=='FFStack' and task=='BinaryClassification':
                        epoch_incorrect_guesses.append(
                        ('input sentence = ' + input_sentence, 'initial stack state = ' + str(stack_states_train[i])))
                else:
                    epoch_incorrect_guesses.append(input_sentence)
            if print_flag == True:
                with open(train_log, 'a') as f:
                    f.write('predicted class = ' + guess + '\n')
                    f.write('actual class = ' + class_category + '\n')

        accuracy = num_correct / len(X_train) * 100

        if (epoch + 1) % 50 == 0:
            print('input layer weights = ', model.fc1.weight)
            print('input layer bias = ', model.fc1.bias)
            print('output layer weights  = ', model.fc2.weight)
            print('output layer bias = ', model.fc2.bias)
            print('input tensor gradient = ', input_tensor.grad)

        print('Accuracy for epoch ', epoch, '=', accuracy, '%')
        all_losses.append(current_loss / len(X_train))
        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)

        accuracies.append(accuracy)

        weights_input_layer.append(model.fc1.weight.clone().detach().numpy())
        if model.freeze_input_layer == False:
            if model.fc1.weight.grad!=None:
                gradients_input_layer.append(model.fc1.weight.grad.clone().detach().numpy())
            elif model.fc1.weight.grad==None:
                gradients_input_layer.append(None)
        elif model.freeze_input_layer == True:
            gradients_input_layer.append(None)
        # if model.fc1.bias==True:
        if model.fc1.bias!=None:
            biases_input_layer.append(model.fc1.bias.clone().detach().numpy())
        else:
            biases_input_layer.append(None)
        weights_output_layer.append(model.fc2.weight.clone().detach().numpy())
        if model.freeze_output_layer == False:
            if model.fc2.weight.grad!=None:
                gradients_output_layer.append(model.fc2.weight.grad.clone().detach().numpy())
            elif model.fc2.weight.grad==None:
                gradients_output_layer.append(None)
        elif model.freeze_output_layer == True:
            gradients_output_layer.append(None)
        if model.fc2.bias==True:
            biases_output_layer.append(model.fc2.bias.clone().detach().numpy())
        else:
            biases_output_layer.append(None)
        confusion_matrices.append(confusion)

        if epoch == num_epochs - 1:
            print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            print('confusion matrix for training set\n', confusion)
            print('Final training accuracy = ', num_correct / len(X_train) * 100, '%')

            if i == len(X_train) - 1:
                print('input tensor = ', input_tensor)

                print('final input sentence = ', input_sentence)
                if model.model_name!='VanillaRNN' and model.model_name!='VanillaGRU' and model.model_name!='VanillaLSTM':
                    print('final stack depth = ', model.stack.stack_depth)
                    print('final false pop count = ', model.stack.false_pop_count)


    df1['epoch'] = epochs
    df1['input layer weights'] = weights_input_layer
    df1['input layer weight gradients'] = gradients_input_layer
    df1['input layer biases'] = biases_input_layer
    df1['output layer weights'] = weights_output_layer
    df1['output layer weight gradients'] = gradients_output_layer
    df1['output layer biases'] = biases_output_layer
    df1['accuracies'] = accuracies
    df1['average epoch losses'] = all_losses
    df1['confusion matrices'] = confusion_matrices
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses

    torch.save(model.state_dict(), modelname)
    torch.save(optimiser.state_dict(), optimname)

    print('all incorrect guesses in training across all epochs = \n', all_epoch_incorrect_guesses)
    return accuracy, df1




def test_model(model, dataset='short'):

    model.eval()
    num_correct = 0
    if dataset=='short':
        num_samples = len(X_test)
        filename = test_log
    elif dataset=='long':
        num_samples = len(X_long)
        filename = long_test_log
    # num_samples = len(X_encoded)
    confusion = torch.zeros(num_classes, num_classes)
    expected_classes = []
    predicted_classes = []
    correct_guesses = []
    incorrect_guesses = []
    print_flag=True
    print('////////////////////////////////////////')
    print('TEST')
    with open(filename,'a') as f:
        f.write('////////////////////////////////////////\n')
        f.write('TEST '+dataset+'\n')
    with torch.no_grad():

        for i in range(num_samples):
            if dataset=='short':
                class_category = y_test_notencoded[i]
                class_tensor = y_test[i]
                input_sentence = X_test_notencoded[i]
                input_tensor = X_test[i]
            elif dataset=='long':
                class_category = y_long_notencoded[i]
                class_tensor=y_long[i]
                input_sentence = X_long_notencoded[i]
                input_tensor = X_long[i]



            if model.model_name=='StackRNN' or model.model_name=='StackLSTM' or model.model_name=='StackGRU' or model.model_name=='RecurrentStack' or model.model_name=='FFStack':
                model.stack.reset()

            if model.model_name =='FFStack':
                model.stack.editStackState(stack_states_test[i][0], stack_states_test[i][1])

            print('////////////////////////////////////////////')
            print('Test sample = ',input_sentence)
            if model.model_name!='VanillaRNN' and model.model_name!='VanillaGRU' and model.model_name!='VanillaLSTM':
                if model.model_name=='FFStack':
                    print('input stack state = ', stack_states_test[i])
                    model.stack.editStackState(stack_states_test[i][0], stack_states_test[i][1])
                print('initial stack state = [', model.stack.stack_depth.item(), ',', model.stack.false_pop_count.item(),
                  ']')

            with open(filename,'a') as f:
                f.write('/////////////////////////////////////////////\n')
                if model.model_name != 'VanillaRNN' and model.model_name != 'VanillaGRU' and model.model_name != 'VanillaLSTM':
                    if model.model_name=='FFStack':
                        f.write('input stack state = '+str(stack_states_test[i])+'\n')
                    f.write('initial stack state = ['+str(model.stack.stack_depth.item())+','+str(model.stack.false_pop_count.item())+
                  ']\n')

            if model.model_name=='VanillaLSTM' or model.model_name=='StackLSTM':
                hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
            elif model.model_name=='VanillaRNN' or model.model_name=='VanillaGRU' or model.model_name=='StackRNN' \
                        or model.model_name=='StackGRU':
                hidden = torch.zeros(1,1,hidden_size)
            if model.model_name=='StackLSTM' or model.model_name=='StackGRU' or model.model_name=='StackRNN' or \
                        model.model_name=='RecurrentStack':
                stack_state = torch.tensor([0,0], dtype=torch.float32)

            for j in range(input_tensor.size()[0]):

                print('input tensor[j][0] = ',input_tensor[j][0])
                with open(filename,'a') as f:
                    f.write('input tensor[j][0] = '+str(input_tensor[j][0])+'\n')


                if model.model_name=='FFStack':
                    output = model(input_tensor[j].squeeze())
                    if print_flag == True:
                        with open(filename, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')
                elif model.model_name == 'VanillaLSTM' or model.model_name=='VanillaRNN' or \
                    model.model_name=='VanillaGRU':
                    output, hidden = model(input_tensor[j].squeeze(), hidden)
                elif model.model_name=='StackLSTM' or model.model_name=='StackRNN' or model.model_name=='StackGRU':
                    output, hidden, stack_state = model(input_tensor[j].squeeze(), hidden, stack_state)
                    if print_flag == True:
                        with open(filename, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')
                elif model.model_name=='RecurrentStack':
                    output, stack_state = model(input_tensor[j].squeeze(), stack_state)
                    if print_flag == True:
                        with open(filename, 'a') as f:
                            f.write('final stack state = [' + str(model.stack.stack_depth.item()) + ',' +
                                        str(model.stack.false_pop_count.item()) + ']\n')
                        print('final stack state = [', model.stack.stack_depth.item(), ',',
                                  model.stack.false_pop_count.item(), ']')




            guess, guess_i = classFromOutput(output)
            class_i = labels.index(class_category)
            print('predicted class = ',guess)
            print('actual class = ',class_category)
            with open(filename,'a') as f:
                f.write('predicted class = '+guess+'\n')
                f.write('actual class = '+class_category+'\n')
            confusion[class_i][guess_i] += 1
            predicted_classes.append(guess_i)
            expected_classes.append(class_i)
            if guess == class_category:
                num_correct+=1
                correct_guesses.append(input_sentence)
            else:
                incorrect_guesses.append(input_sentence)


    accuracy = num_correct/num_samples*100
    print('confusion matrix for test set \n',confusion)
    conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
    print('correct guesses in testing = ', correct_guesses)
    print('incorrect guesses in testing = ', incorrect_guesses)

    with open(filename,'a') as f:
        f.write('test accuracy = '+str(accuracy)+'%\n')
        f.write('confusion matrix for test set = \n'+str(confusion)+'\n')
        f.write('correct guesses in testing = '+str(correct_guesses)+'\n')
        f.write('incorrect guesses in testing = '+str(incorrect_guesses)+'\n')
    return accuracy

# train_accuracy, df = train_model(model)
# print(train_accuracy)
# test_accuracy = test_model(model, 'short')

with open(file_name,'a') as f:
    f.write('Output activation = '+output_activation+'\n')
    f.write('Frozen layers = '+frozen_layers+'\n')
    f.write('Initialisation = '+initialisation+'\n')
    f.write('Optimiser used = '+use_optimiser+'\n')
    f.write('Learning rate = '+str(learning_rate)+'\n')
    f.write('Number of runs = '+str(num_runs)+'\n')
    f.write('Number of epochs in each run = '+str(num_epochs)+'\n')
    f.write('Saved model name = '+modelname+'\n')
    f.write('Saved optimiser name = '+optimname+'\n')
    f.write('Excel name = '+excel_name+'\n')
    f.write('Train log name = '+train_log+'\n')
    f.write('Test log name = ' +test_log + '\n')
    f.write('Long test log name = ' + long_test_log + '\n')
    f.write('///////////////////////////////////////////////////////////////\n')
    f.write('\n')


for i in range(num_runs):
    if model_name == 'RecurrentStack':
        model = RecurrentStack(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size,
                               freeze_input_layer,
                               freeze_output_layer, output_activation, initialisation, task, bias)
    elif model_name == 'FFStack':
        model = FFStack(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size, freeze_input_layer,
                        freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'StackLSTM':
        model = StackLSTM(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size, freeze_input_layer,
                          freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'StackGRU':
        model = StackGRU(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size, freeze_input_layer,
                         freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'StackRNN':
        model = StackRNN(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size, freeze_input_layer,
                         freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'VanillaLSTM':
        model = VanillaLSTM(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size,
                            freeze_input_layer,
                            freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'VanillaRNN':
        model = VanillaRNN(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size,
                           freeze_input_layer,
                           freeze_output_layer, output_activation, initialisation, task)
    elif model_name == 'VanillaGRU':
        model = VanillaGRU(input_size, hidden_size, output_size, num_layers, stack_input_size, stack_output_size,
                           freeze_input_layer,
                           freeze_output_layer, output_activation, initialisation, task)

    if model.model_name!='VanillaRNN' and model.model_name!='VanillaGRU' and model.model_name!='VanillaLSTM':
        model.stack.editStackState(0, 0)
    # train_accuracy = train_model(model)
    train_accuracy, df = train_model(model)
    train_accuracies.append(train_accuracy)
    train_dataframes.append(df)
    test_accuracy = test_model(model)
    test_accuracies.append(test_accuracy)
    if model.model_name!='FFStack':
        long_test_accuracy = test_model(model,'long')
        long_test_accuracies.append(long_test_accuracy)
    elif model.model_name=='FFStack':
        long_test_accuracies.append(0)

    with open(file_name, "a") as f:
        f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
        f.write('test accuracy for run ' + str(i) + ' = ' + str(test_accuracy) + '%\n')
        if model.model_name!='FFStack':
            f.write('long test accuracy for run ' + str(i) + ' = ' + str(long_test_accuracy) + '%\n')

runs = []
for i in range(len(train_dataframes)):
    runs.append('run'+str(i))

dfs = dict(zip(runs,train_dataframes))
writer = pandas.ExcelWriter(excel_name,engine='xlsxwriter')

for sheet_name in dfs.keys():
    dfs[sheet_name].to_excel(writer,sheet_name=sheet_name,index=False)

writer.save()



max_train_accuracy = max(train_accuracies)
min_train_accuracy = min(train_accuracies)
avg_train_accuracy = sum(train_accuracies)/len(train_accuracies)
std_train_accuracy = np.std(train_accuracies)


max_test_accuracy = max(test_accuracies)
min_test_accuracy = min(test_accuracies)
avg_test_accuracy = sum(test_accuracies)/len(test_accuracies)
std_test_accuracy = np.std(test_accuracies)

if model_name!='FFStack':
    max_long_test_accuracy = max(long_test_accuracies)
    min_long_test_accuracy = min(long_test_accuracies)
    avg_long_test_accuracy = sum(long_test_accuracies) / len(test_accuracies)
    std_long_test_accuracy = np.std(long_test_accuracies)

with open(file_name, "a") as f:
    f.write('/////////////////////////////////////////////////////////////////\n')
    f.write('Maximum train accuracy = '+str(max_train_accuracy)+'%\n')
    f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
    f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
    f.write('Standard Deviation for train accuracy = '+str(std_train_accuracy)+'\n')
    f.write('/////////////////////////////////////////////////////////////////\n')
    f.write('Maximum test accuracy = ' + str(max_test_accuracy) + '%\n')
    f.write('Minimum test accuracy = ' + str(min_test_accuracy) + '%\n')
    f.write('Average test accuracy = ' + str(avg_test_accuracy) + '%\n')
    f.write('Standard Deviation for test accuracy = ' + str(std_test_accuracy) + '\n')
    if model_name!='FFStack':
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum long test accuracy = ' + str(max_long_test_accuracy) + '%\n')
        f.write('Minimum long test accuracy = ' + str(min_long_test_accuracy) + '%\n')
        f.write('Average long test accuracy = ' + str(avg_long_test_accuracy) + '%\n')
        f.write('Standard Deviation for long test accuracy = ' + str(std_long_test_accuracy) + '\n')


if model_name!='FFStack':
    width = 0.3
    plt.bar(np.arange(len(train_accuracies)), train_accuracies, width=width, label='Train Accuracy')
    plt.bar(np.arange(len(test_accuracies))+width, test_accuracies, width=width, label='Test Accuracy')
    plt.bar(np.arange(len(long_test_accuracies)) + (2*width), long_test_accuracies, width=width, label='Long Test Accuracy')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(len(train_accuracies)))
elif model_name=='FFStack':
    width = 0.3
    plt.bar(np.arange(len(train_accuracies)), train_accuracies, width=width, label='Train Accuracy')
    plt.bar(np.arange(len(test_accuracies)) + width, test_accuracies, width=width, label='Test Accuracy')
    plt.bar(np.arange(len(long_test_accuracies)) + (2*width), long_test_accuracies, width=width, label='Long Test Accuracy')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
plt.xticks(np.arange(len(train_accuracies)))
plt.yticks(np.arange(0, 101, step=10))
plt.ylabel('Accuracy (%)')
plt.xlabel('Run Number')

plt.savefig(plot_name)
plt.show()