import numpy as np
import os
import torch

def compute_apfd(true_labels, predicted_labels):
    n = len(true_labels)

    potential_faults = []
    fault_positions = []
    
    for i, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        true_label = true_label[0]
        predicted_label = predicted_label[0]
        if true_label != predicted_label and [true_label, predicted_label] not in potential_faults:
            fault_positions.append(i)
            potential_faults.append([true_label, predicted_label])

    num_faults = len(potential_faults)
    print(num_faults)
    apfd = 1 - (sum(fault_positions) / (num_faults * n)) + 1 / (2 * n)
    return apfd * 100

dataset = 'cifar10'
model = 'resnet20'
pred_labels = np.flip(np.load('./neuron_sensitive_samples_benign_' + dataset + '_' + model + '_pred_labels.npy')[:,None])[:500]
true_labels = np.flip(np.load('./neuron_sensitive_samples_benign_' + dataset + '_' + model + '_labels.npy'))[:500]
print(pred_labels.shape, true_labels.shape)
print('APFD:', compute_apfd(true_labels, pred_labels))

