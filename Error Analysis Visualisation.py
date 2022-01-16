import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from pathlib import Path
import torch
import numpy as np


##############################
# CONFIGURE MODEL AND FILES TO READ

# Stack models
# model_name = 'RecurrentStack'
model_name = 'FFStack'

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

# freeze_output_layer=False
freeze_output_layer=True

initialisation='Random'
# initialisation='Correct'

use_optimiser='Adam'
# use_optimiser='SGD'

learning_rate=0.001
num_runs=10
# num_epochs=1000
num_epochs=10000

bias=False
# bias=True

task = 'BinaryClassification'
# task='StackStateOutput'

if model_name!='FFStack' and model_name!='RecurrentStack':
    task='BinaryClassification'

if model_name=='FFStack':
    folder_name = "LOGS - Dyck1 Binary Classification FFStack Models/"
elif model_name=='RecurrentStack':
    folder_name='LOGS - Dyck1 Binary Classification Recurrent Stack Models'
elif model_name=='StackRNN' or model_name=='StackLSTM' or model_name=='StackGRU':
    folder_name='LOGS - Dyck1 Binary Classification Mixed Models'


###############################################################################

frozen_layers = 'no_frozen_layers'
if freeze_input_layer==True and freeze_output_layer==False:
    frozen_layers='frozen_input_layer'
elif freeze_input_layer==False and freeze_output_layer==True:
    frozen_layers='frozen_output_layer'






# excel_name = folder_name+'/Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+str(num_layers)+'layers_'+str(hidden_size)+'hidden_units_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+'bias_'+str(bias)+'_'+frozen_layers+'.xlsx'
if model_name=='FFStack':
    excel_name = r""+folder_name+'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_' + output_activation + '_'+ initialisation + '_initialisation_' + use_optimiser + '_lr=' + str(
        learning_rate) + '_' + str(num_epochs) + 'epochs_' + frozen_layers + '.xlsx'
    loss_acc_fig_name = 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_' + output_activation + '_'+ initialisation + '_initialisation_' + use_optimiser + '_lr=' + str(
        learning_rate) + '_' + str(num_epochs) + 'epochs_' + frozen_layers + '_training losses and accuracies.jpg'
    weights_gradients_fig_name = 'Dyck1_' + task + '_' + str(
        num_bracket_pairs) + '_bracket_pairs_' + model_name + '_' + output_activation + '_' + initialisation + '_initialisation_' + use_optimiser + '_lr=' + str(
        learning_rate) + '_' + str(num_epochs) + 'epochs_' + frozen_layers + '_weights and gradients.png'


runs = [i for i in range(10)]
print(runs)
print(excel_name)


dfs = []
accuracy_plots = []
loss_plots = []
plots = []
# fig, axs = plt.subplots(5, 2, figsize=(18,25), constrained_layout=True)
# # plt.rcParams['figure.figsize']=(10,8)
#
# # f = plt.figure()
# # f.set_figheight(10)
#
# for i in range(10):
#     print(i)
#     dfs.append(pd.read_excel(excel_name,sheet_name='run'+str(runs[i])))
#     # fig, axs = plt.subplots(1, 2)
#     # plt.show()
#     if i%2==0:
#         axs[i//2,0].plot(dfs[i]['epoch'], dfs[i]['accuracies']/100, label='accuracies')
#         axs[i//2,0].set_title('train run '+str(i)+'\n losses and accuracies')
#         axs[i//2,0].set_yticks([0,0.25,0.5,0.75,1.00])
#     elif i%2!=0:
#         axs[i//2, 1].plot(dfs[i]['epoch'], dfs[i]['accuracies'] / 100, label='accuracies')
#         axs[i//2, 1].set_title('train run ' + str(i)+'\n losses and accuracies')
#         axs[i//2, 1].set_yticks([0, 0.25, 0.5, 0.75, 1.00])
#     # losses=dfs[i]['average epoch losses']
#     losses=[]
#     # for loss in losses:
#     for loss in dfs[i]['average epoch losses']:
#         start = loss.index('(')
#         end = loss.index(',')
#         loss = loss[:end]
#         loss = loss[start+1:]
#         losses.append(float(loss))
#         # loss = int(loss)
#     dfs[i]['losses'] = losses
#     # ax2.plot(dfs[i]['epoch'], dfs[i]['average epoch losses'])
#     if i%2==0:
#         axs[i//2, 0].plot(dfs[i]['epoch'],dfs[i]['losses'], label='losses')
#         axs[i//2, 0].legend()
#     elif i%2!=0:
#         axs[i//2,1].plot(dfs[i]['epoch'],dfs[i]['losses'], label='losses')
#         axs[i//2,1].legend()
#     # axs[1].plot(dfs[i]['epoch'],dfs[i]['losses'])
#     # axs[1].set_title('losses across epochs for run '+str(i))
#     # axs[1].set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
#     # fig.suptitle('Dyck1_' + task + '_'+ model_name + '_' + output_activation + '_'+ initialisation + '_initialisation_' + frozen_layers)
# plt.show()
# plt.savefig(loss_acc_fig_name)


fig, axs = plt.subplots(10, 5, figsize=(50,50), constrained_layout=True)
# plt.rcParams['figure.figsize']=(10,8)

# f = plt.figure()
# f.set_figheight(10)

for i in range(10):
    print(i)
    dfs.append(pd.read_excel(excel_name,sheet_name='run'+str(runs[i])))
    # fig, axs = plt.subplots(1, 2)
    # plt.show()

    axs[i,0].plot(dfs[i]['epoch'], dfs[i]['accuracies']/100, label='accuracies')
    axs[i,0].set_title('train run '+str(i)+'\n losses and accuracies')
    axs[i,0].set_yticks([0,0.1, 0.2,0.3,0.4, 0.5, 0.6,0.7,0.8, 0.9,1.00])


    # losses=dfs[i]['average epoch losses']
    losses=[]
    # for loss in losses:
    for loss in dfs[i]['average epoch losses']:
        start = loss.index('(')
        end = loss.index(',')
        loss = loss[:end]
        loss = loss[start+1:]
        losses.append(float(loss))
        # loss = int(loss)
    dfs[i]['losses'] = losses
    # ax2.plot(dfs[i]['epoch'], dfs[i]['average epoch losses'])

    axs[i, 0].plot(dfs[i]['epoch'],dfs[i]['losses'], label='losses')
    axs[i, 0].legend()

    input_weight_00 = []
    input_weight_01=[]
    input_weight_10=[]
    input_weight_11=[]
    for input_weight in dfs[i]['input layer weights']:
        input_weight=input_weight.replace('[[ ','')
        # input_weight=input_weight.replace(' ','')
        input_weight = input_weight.replace('[  ', '')
        input_weight = input_weight.replace('[ ','')
        input_weight = input_weight.replace(' ]', '')
        input_weight = input_weight.replace('[','')
        input_weight = input_weight.replace(']', '')
        input_weight=input_weight.replace('\n',' ')
        input_weight=input_weight.replace('   ', '  ')
        input_weight=input_weight.replace('  ',' ')
        input_weight = input_weight.replace('][', ' ')
        input_weight=input_weight.replace('   ', '  ')
        input_weight=input_weight.replace('  ',' ')
        input_weight=input_weight.replace('  ', ' ')
        input_weight = input_weight.lstrip()

        # if input_weight.split(' ')[0]==' ':
        #     input
        # print(input_weight)
        # print(input_weight.split(' ')[3])
        # print(input_weight.split(' ')[0])
        # print(float(input_weight.split(' ')[0].replace(' ','')))
        # print(input_weight.split(' ')[1])
        # print(float(input_weight.split(' ')[1].replace(' ','')))
        # print(input_weight.split(' ')[0])
        # print(input_weight.split(' ')[3])

        input_weight_00.append(float(input_weight.split(' ')[0].replace(' ','')))
        input_weight_01.append(float(input_weight.split(' ')[1]))
        input_weight_10.append(float(input_weight.split(' ')[2]))
        input_weight_11.append(float(input_weight.split(' ')[3].lstrip()))

    # print('input weight 00\n',input_weight_00[:5])
    # print('input weight 01\n', input_weight_01[:5])
    # print('input weight 10\n', input_weight_10[:5])
    # print('input weight 11\n', input_weight_11[:5])

    dfs[i]['input weight 00']=input_weight_00
    dfs[i]['input weight 01'] = input_weight_01
    dfs[i]['input weight 10'] = input_weight_10
    dfs[i]['input weight 11'] = input_weight_11
    axs[i,1].plot(dfs[i]['epoch'],dfs[i]['input weight 00'], label='input weight 00')
    axs[i, 1].plot(dfs[i]['epoch'], dfs[i]['input weight 01'], label='input weight 01')
    axs[i, 1].plot(dfs[i]['epoch'], dfs[i]['input weight 10'], label='input weight 10')
    axs[i, 1].plot(dfs[i]['epoch'], dfs[i]['input weight 11'], label='input weight 11')
    axs[i,1].legend()
    axs[i, 1].set_title('train run ' + str(i) + '\n input weights')

    input_weight_grad_00 = []
    input_weight_grad_01 = []
    input_weight_grad_10 = []
    input_weight_grad_11 = []
    for input_weight_grad in dfs[i]['input layer weight gradients']:
        # if type(input_weight_grad) is str:
        #     input_weight_grad = input_weight_grad.replace('[ ', '')
        #     input_weight_grad = input_weight_grad.replace(' ]', '')
        #     input_weight_grad = input_weight_grad.replace('][', ' ')
        #     input_weight_grad = input_weight_grad.replace('[', '')
        #     input_weight_grad = input_weight_grad.replace(']', '')
        #     input_weight_grad = input_weight_grad.replace('\n', ' ')
        #     input_weight_grad = input_weight_grad.replace('   ', '  ')
        #     input_weight_grad = input_weight_grad.replace('  ', ' ')
        #     input_weight_grad = input_weight_grad.replace('][', ' ')
        #     input_weight_grad = input_weight_grad.replace('   ', '  ')
        #     input_weight_grad = input_weight_grad.replace('  ', ' ')
        #     input_weight_grad = input_weight_grad.replace('  ', ' ')
        # # print(input_weight_grad)
        # # print(input_weight_grad.split(' ')[0])
        # # print(float(input_weight_grad.split(' ')[0]))
        # # print(input_weight_grad.split(' ')[1])
        # # print(float(input_weight_grad.split(' ')[1].replace(' ','')))
        # # print(input_weight_grad.split(' ')[0])
        #
        #     input_weight_grad_00.append(float(input_weight_grad.split(' ')[0]))
        #     input_weight_grad_01.append(float(input_weight_grad.split(' ')[1]))
        #     input_weight_grad_10.append(float(input_weight_grad.split(' ')[2]))
        #     input_weight_grad_11.append(float(input_weight_grad.split(' ')[3]))
        if freeze_input_layer==True:
            input_weight_grad_00.append(0.0)
            input_weight_grad_01.append(0.0)
            input_weight_grad_10.append(0.0)
            input_weight_grad_11.append(0.0)
        elif freeze_input_layer==False:
        # print(input_weight_grad)
            input_weight_grad = input_weight_grad.replace('[ ', '')
            input_weight_grad = input_weight_grad.replace(' ]', '')
            input_weight_grad = input_weight_grad.replace('][', ' ')
            input_weight_grad = input_weight_grad.replace('[', '')
            input_weight_grad = input_weight_grad.replace(']', '')
            input_weight_grad = input_weight_grad.replace('\n', ' ')
            input_weight_grad = input_weight_grad.replace('   ', '  ')
            input_weight_grad = input_weight_grad.replace('  ', ' ')
            input_weight_grad = input_weight_grad.replace('][', ' ')
            input_weight_grad = input_weight_grad.replace('   ', '  ')
            input_weight_grad = input_weight_grad.replace('  ', ' ')
            input_weight_grad = input_weight_grad.replace('  ', ' ')
            # print(input_weight_grad)
            # print(input_weight_grad.split(' ')[0])
            # print(float(input_weight_grad.split(' ')[0]))
            # print(input_weight_grad.split(' ')[1])
            # print(float(input_weight_grad.split(' ')[1].replace(' ','')))
            # print(input_weight_grad.split(' ')[0])

            input_weight_grad_00.append(float(input_weight_grad.split(' ')[0]))
            input_weight_grad_01.append(float(input_weight_grad.split(' ')[1]))
            input_weight_grad_10.append(float(input_weight_grad.split(' ')[2]))
            input_weight_grad_11.append(float(input_weight_grad.split(' ')[3]))

    # print(len(input_weight_grad_00))
    # print('input weight 00\n', input_weight_00[:5])
    # print('input weight 01\n', input_weight_01[:5])
    # print('input weight 10\n', input_weight_10[:5])
    # print('input weight 11\n', input_weight_11[:5])

    dfs[i]['input weight grad 00'] = input_weight_grad_00
    dfs[i]['input weight grad 01'] = input_weight_grad_01
    dfs[i]['input weight grad 10'] = input_weight_grad_10
    dfs[i]['input weight grad 11'] = input_weight_grad_11
    axs[i, 2].plot(dfs[i]['epoch'], dfs[i]['input weight grad 00'], label='input grad 00')
    axs[i, 2].plot(dfs[i]['epoch'], dfs[i]['input weight grad 01'], label='input grad 01')
    axs[i, 2].plot(dfs[i]['epoch'], dfs[i]['input weight grad 10'], label='input grad 10')
    axs[i, 2].plot(dfs[i]['epoch'], dfs[i]['input weight grad 11'], label='input grad 11')
    axs[i, 2].legend()
    axs[i, 2].set_title('train run ' + str(i) + '\n input weight gradients')

    output_weight_0 = []
    output_weight_1 = []

    for output_weight in dfs[i]['output layer weights']:
        output_weight = output_weight.replace('[ ', '')
        output_weight = output_weight.replace(' ]', '')
        output_weight = output_weight.replace('][', ' ')
        output_weight = output_weight.replace('[', '')
        output_weight = output_weight.replace(']', '')
        output_weight = output_weight.replace('\n', ' ')
        output_weight = output_weight.replace('   ', '  ')
        output_weight = output_weight.replace('  ', ' ')
        output_weight = output_weight.replace('][', ' ')
        output_weight = output_weight.replace('   ', '  ')
        output_weight = output_weight.replace('  ', ' ')
        output_weight_0.append(float(output_weight.split(' ')[0]))
        output_weight_1.append(float(output_weight.split(' ')[1]))



    dfs[i]['output weight 0'] = output_weight_0
    dfs[i]['output weight 1'] = output_weight_1

    axs[i, 3].plot(dfs[i]['epoch'], dfs[i]['output weight 0'], label='output weight 0')
    axs[i, 3].plot(dfs[i]['epoch'], dfs[i]['output weight 1'], label='output weight 1')

    axs[i, 3].legend()
    axs[i, 3].set_title('train run ' + str(i) + '\n output weights')

    output_weight_grad_0 = []
    output_weight_grad_1 = []

    for output_weight_grad in dfs[i]['output layer weight gradients']:
        # print(output_weight_grad)
        if freeze_output_layer==True:
            output_weight_grad_0.append(0.0)
            output_weight_grad_1.append(0.0)

        elif freeze_output_layer==False:

            output_weight_grad = output_weight_grad.replace('[[ ', '')
            output_weight_grad = output_weight_grad.replace('[', '')
            output_weight_grad = output_weight_grad.replace(']', '')
            output_weight_grad = output_weight_grad.replace('][', ' ')
            output_weight_grad=output_weight_grad.replace('[[ ', '')
            output_weight_grad = output_weight_grad.replace('[ ', '')
            output_weight_grad = output_weight_grad.replace(' ]', '')
            output_weight_grad = output_weight_grad.replace('\n', ' ')
            output_weight_grad = output_weight_grad.replace('   ', '  ')
            output_weight_grad = output_weight_grad.replace('  ', ' ')
            output_weight_grad = output_weight_grad.replace('][', ' ')
            output_weight_grad = output_weight_grad.replace('   ', '  ')
            output_weight_grad = output_weight_grad.replace('  ', ' ')
            output_weight_grad = output_weight_grad.replace('  ', ' ')
            # print(output_weight_grad)
            # print(output_weight_grad.split(' '))
            # print(output_weight_grad.split(' ')[0])
            output_weight_grad_0.append(float(output_weight_grad.split(' ')[0]))
            output_weight_grad_1.append(float(output_weight_grad.split(' ')[1]))

    dfs[i]['output weight grad 0'] = output_weight_grad_0
    dfs[i]['output weight grad 1'] = output_weight_grad_1

    axs[i, 4].plot(dfs[i]['epoch'], dfs[i]['output weight grad 0'], label='output grad 0')
    axs[i, 4].plot(dfs[i]['epoch'], dfs[i]['output weight grad 1'], label='output grad 1')

    axs[i, 4].legend()
    axs[i, 4].set_title('train run ' + str(i) + '\n output weight gradients')

plt.show()
# plt.savefig(weights_gradients_fig_name)

# print(dfs[4].head())
# print(dfs[4].columns)
# print(len(dfs))

# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot()


# 1 clipping correct fully trainable
# 2 clipping random fully trainable
# 3 clipping random frozen input
# 4 clipping random frozen output
# 5 sigmoid correct fully trainable
# 6 sigmoid correct frozen inputt
# 7 sigmoid correct frozen output
# 8 sigmoid random fully trainable
# 9 sigmoid random frozen inputt
# 10 sigmoid random frozen outputt