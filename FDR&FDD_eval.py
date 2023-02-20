import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
# from feature_wise_pgd import PGD
import argparse
import random
# from PGD import PGD
# from mcmc import MCMC
from resnet20 import ResNet20
from LeNet import LeNet5
from benign_perturbations import benign_aug
# from deepgini import deep_metric, intersection

import torchattacks



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('-p', '--percentage', type=int, default=20, help='Top k percentage of selected cases')
    args = parser.parse_args()

    dataset = 'cifar10'
    model = 'resnet20'
    # Dataset
    eval_images_path = './neuron_sensitive_samples_' + dataset + '_' + model + '_images.npy'
    eval_labels_path = './neuron_sensitive_samples_' + dataset + '_' + model + '_labels.npy'
    testmodel = torch.load('./pretrained/cifar10_resnet20_0.861.pt').cuda()
    np.random.seed(123)

    test_x_numpy = np.load(eval_images_path)
    test_y_numpy = np.load(eval_labels_path)

    total_samples_num = test_y_numpy.shape[0]
    samples_num = int((args.percentage / 20) * total_samples_num)

    x_test=torch.from_numpy(test_x_numpy)
    y_test=torch.from_numpy(test_y_numpy)
    x_test = torch.flip(x_test, dims=[0])[:samples_num]
    y_test = torch.flip(y_test, dims=[0])[:samples_num]
    test_dataset=TensorDataset(x_test,y_test)


    batch_size = 100
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    testmodel.eval()

    batch_idx = int((total_samples_num * args.percentage) / (100 * batch_size))

    acc_array = []
    

    all_correct_num = 0
    all_sample_num = 0
    FDR = {}
    total_FDR_list = []
    FDR_list = []
    gini_score_vec = []
    # pred = []
    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):

            test_x, test_label = test_x.cuda(), test_label.cuda()

            predict_y_adv, _ = testmodel(test_x)
            # pred.append(predict_y_adv.detach().cpu().numpy())


            predict_y_adv = predict_y_adv.argmax(dim=1, keepdim=False)
            current_correct_num = predict_y_adv == test_label.squeeze()
            for i in range(len(predict_y_adv)):
                if predict_y_adv[i]!=test_label[i]:
                    FDR[(str(predict_y_adv[i]),str(test_label[i]))] = 1
            total_FDR_list.append(len(FDR))
           

            all_correct_num += current_correct_num.sum().item()
            all_sample_num += test_label.shape[0]
            acc = current_correct_num.sum().item() / test_x.shape[0]
            acc_array.append(acc)

    # print(dataset,model,all_correct_num)
    # print(np.array(pred).shape)
    # _, gini_score = deep_metric(np.array(pred).reshape(2000,10))
    # print(dataset,model,'Gini Score: ', gini_score / 2000)
    print(dataset,model,'Selected cases info: ')
    print(dataset,model,'Fault Detection Diversity: ',total_FDR_list)
    print(dataset,model,'AUC: ', np.array(total_FDR_list).sum() / (90*20))
    print(dataset,model,'Fault Detection Rate: ', 1 - np.mean(acc_array))
    print(dataset,model,'Acc: ', np.mean(acc_array))
    print(dataset,model,'Total Correct Num: ', np.sum(acc_array) * batch_size)
