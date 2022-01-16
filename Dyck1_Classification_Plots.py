import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


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
hidden_size=2
stack_input_size=2
stack_output_size=2
output_size = 1

# output_activation='Clipping'
output_activation='Sigmoid'

freeze_input_layer=False
# freeze_input_layer=True

freeze_output_layer=False
# freeze_output_layer=True

initialisation='Random'
# initialisation='Correct'

use_optimiser='Adam'
# use_optimiser='SGD'

learning_rate=0.001
num_runs=10
num_epochs=1000
# num_epochs=10000

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



plot_name = 'Dyck1_'+task+'_'+str(num_bracket_pairs)+'_bracket_pairs_'+model_name+'_'+output_activation+'_'+initialisation+'_initialisation_'+use_optimiser+'_lr='+str(learning_rate)+'_'+str(num_epochs)+'epochs_'+frozen_layers+'_PLOT.png'

acc_train = []
acc_test = []
acc_length = []

if plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Random_initialisation_Adam_lr=0.001_10000epochs_no_frozen_layers_PLOT.png':
    acc_train = [75.0, 75.0, 75.0, 100.0, 100.0, 75.0, 75.0, 75.0, 75.0, 75.0]
    acc_test = [50.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    acc_length=[0,0,0,0,0,0,0,0,0,0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Random_initialisation_Adam_lr=0.001_10000epochs_frozen_input_layer_PLOT.png':
    acc_train=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Random_initialisation_Adam_lr=0.001_10000epochs_frozen_output_layer_PLOT.png':
    acc_train = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0]
    acc_test = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Clipping_Correct_initialisation_Adam_lr=0.001_10000epochs_no_frozen_layers_PLOT.png':
    acc_train=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Clipping_Random_initialisation_Adam_lr=0.001_10000epochs_frozen_input_layer_PLOT.png':
    acc_train = [50, 100, 100, 100, 50, 100, 50, 50, 50, 100]
    acc_test = [50, 100, 50, 100, 50, 100, 50, 50, 50, 100]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Clipping_Random_initialisation_Adam_lr=0.001_10000epochs_frozen_output_layer_PLOT.png':
    acc_train = [100, 100, 100, 100, 100, 100, 50, 50, 100, 100]
    acc_test = [100, 100, 100, 100, 100, 100, 50, 50, 100, 100]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Clipping_Random_initialisation_Adam_lr=0.001_10000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100, 50, 100, 50, 100, 100, 100, 100, 100, 75]
    acc_test = [100, 50, 100, 50, 100, 100, 100, 100, 50, 100]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_10000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_10000epochs_frozen_output_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_FFStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_10000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Clipping_Correct_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length= [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Clipping_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [75, 100, 100, 95.83, 100, 50, 95.83, 50, 50, 100]
    acc_test = [54.54, 100, 100, 63.63, 100, 54.54, 63.63, 54.54, 54.54, 100]
    acc_length = [69.88, 100, 100, 89.76, 100, 50, 89.76, 50, 50, 100]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Clipping_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_output_layer_PLOT.png':
    acc_train = [100, 100, 100, 100, 50, 100, 100, 50, 50, 50]
    acc_test = [100, 100, 100, 100, 36.36, 100, 100, 36.36, 36.36, 36.35]
    acc_length = [100, 100, 100, 100, 49.94, 100, 100, 49.94, 49.94, 49.94]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Clipping_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [75, 91.67, 100, 100, 75, 100, 83.33, 100, 100, 50]
    acc_test = [63.63, 100, 100, 100, 63.63, 100, 100, 100, 100,54.54]
    acc_length = [50.86, 100, 100, 100, 50.85, 100, 100, 100, 100, 50]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_1000epochs_frozen_output_layer_PLOT.png':
    acc_train = [70.83, 70.83, 70.83, 70.83, 70.83, 70.83, 70.83, 70.83, 70.83, 70.83]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Correct_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5]
    acc_test = [36.36, 36.36, 36.36, 36.36, 36.36, 36.36, 36.36, 36.36, 36.36, 36.36]
    acc_length = [49.94, 49.94, 49.94, 49.94, 49.94, 49.94, 49.94, 49.94, 49.94, 49.94]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_length = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_output_layer_PLOT.png':
    acc_train = [75, 66.67, 66.67, 75, 75, 75, 79.17, 70.83, 70.83, 75]
    acc_test = [45.45, 100, 100, 100, 100, 45.45, 100, 45.45, 45.45, 100]
    acc_length = [49.94, 100, 100, 100, 100, 49.94, 100, 49.94, 49.94, 100]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_RecurrentStack_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [50, 66.67, 45.83, 50, 50, 50, 50, 45.83, 70.83, 45.83]
    acc_test = [54.54, 45.45, 36.36, 54.54, 54.54, 54.54, 54.54, 36.36,100, 36.36]
    acc_length = [50, 49.94, 49.94, 50, 50, 50, 50, 49.94, 100, 49.94]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackGRU_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100, 100, 100, 100, 75, 100, 100, 100, 100, 100]
    acc_test = [81.81, 81.81, 81.81, 81.81, 54.54, 81.81, 81.81, 81.81, 81.81, 81.81]
    acc_length = [51.67, 50.86, 83.34, 61.29, 54.79, 56.95, 57.35, 58.98, 51.18, 50.85]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackGRU_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [95.83, 95.83, 100, 100, 95.83, 100, 95.83, 100, 100, 100]
    acc_test = [90.91, 100, 100, 100, 90.91, 90.91, 90.91, 100, 90.91, 100]
    acc_length = [92.99, 88.40, 52.05, 83.46, 49.37, 65.98, 56.6, 52.88, 50, 43.72]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackLSTM_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    acc_test = [90.91, 90.91, 90.91, 90.91, 100, 100, 90.91, 90.91, 100, 90.91]
    acc_length = [93.69, 94.84, 65.625, 98.76, 100, 97.17, 96.25, 97.01, 98.76]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackLSTM_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [90.91, 90.91, 100, 90.91, 90.91, 81.82, 90.91, 100, 90.91, 90.91]
    acc_length = [64.96, 96.04, 50, 65.93, 68.997, 50, 97.16, 70.48, 69.26, 78.125]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackRNN_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    acc_test = [72.73, 72.73, 81.82, 72.73, 72.73, 72.73, 72.73, 81.82, 90.91, 72.73]
    acc_length = [61.797, 56.6, 50.44, 51.76, 65.05, 60.74, 65.05, 51.58, 67.6, 66.2]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_StackRNN_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [72.73, 72.73, 72.73, 81.82, 81.82, 72.73, 72.73, 72.73, 72.73, 72.73]
    acc_length = [60.47, 60.47, 60.47, 60.47, 51.42, 62.42, 60.47, 60.47, 60.47, 60.47]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaGRU_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [100, 100, 100, 90.91, 90.91, 90.91, 100, 100, 100, 100]
    acc_length = [63.79, 53.96, 49.25, 59.49, 51.08, 48.58, 57.9, 70.96, 49.73, 67.31]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaGRU_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100, 100, 100, 100, 100, 100, 100, 100, 100, 95.83]
    acc_test = [81.82, 72.73, 81.82, 90.91, 81.82, 81.82, 81.82, 81.82, 81.82, 81.82]
    acc_length = [59.07, 75.57, 53.02, 60.39, 50.72, 53.89, 45.65, 58.36, 56.59, 61.72]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaLSTM_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    acc_test = [90.91, 90.91, 90.91, 90.91, 100, 90.91, 90.91, 100, 100, 63.64]
    acc_length = [53.32, 52.25, 90.16, 50.28, 49.34, 54.86, 52.099, 56.41, 80.19, 67.52]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaLSTM_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100, 100, 100, 100, 100, 95.83, 100, 100, 100, 100]
    acc_test = [100, 100, 90.91, 90.91, 100, 81.82, 100, 81.82, 90.91, 90.91]
    acc_length = [77.49, 62.24, 64.37, 72.796, 74.13, 64.27, 47.67, 57.82, 49.15, 85.68]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaRNN_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_frozen_input_layer_PLOT.png':
    acc_train = [87.5, 95.83, 95.83, 95.83, 95.83, 100, 100, 95.83, 95.83, 95.83]
    acc_test = [81.82, 81.82, 81.82, 81.82, 81.82, 90.91, 81.82, 72.73, 81.82, 81.82]
    acc_length = [59.04, 60.47, 61.08, 61.19, 60.68, 76.01, 69.02, 58.14, 60.47, 61.21]
elif plot_name=='Dyck1_BinaryClassification_2_bracket_pairs_VanillaRNN_Sigmoid_Random_initialisation_Adam_lr=0.001_1000epochs_no_frozen_layers_PLOT.png':
    acc_train = [100, 87.5, 95.83, 95.83, 91.67, 95.83, 95.83, 100, 87.5, 87.5]
    acc_test = [81.82, 90.91, 100, 100, 81.82, 90.91, 100, 100, 90.91, 90.91]
    acc_length = [82.93, 58.67, 89.2, 91.32, 59.91, 57.86, 61.13, 46.22, 56.16, 58.64]

if model_name!='FFStack':
    width = 0.3
    plt.bar(np.arange(len(acc_train)), acc_train, width=width, label='Train Accuracy')
    plt.bar(np.arange(len(acc_test))+width, acc_test, width=width, label='Test Accuracy')
    plt.bar(np.arange(len(acc_length)) + (2*width), acc_length, width=width, label='Long Test Accuracy')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(len(acc_train)))
elif model_name=='FFStack':
    width = 0.3
    plt.bar(np.arange(len(acc_train)), acc_train, width=width, label='Train Accuracy')
    plt.bar(np.arange(len(acc_test)) + width, acc_test, width=width, label='Test Accuracy')
    plt.bar(np.arange(len(acc_length)) + (2*width), acc_length, width=width, label='Long Test Accuracy')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
plt.xticks(np.arange(len(acc_train)))
plt.yticks(np.arange(0, 101, step=10))
plt.ylabel('Accuracy (%)')
plt.xlabel('Run Number')

plt.savefig(plot_name)
plt.show()