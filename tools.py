import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import MLPNet, CNN_small, CNN, NewsNet, LeNet, LeNet_bayes
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.utils.data as Data
import argparse, sys
import numpy as np
import transformer
from tqdm import tqdm
import datetime
import data
import copy
import resnet
import resnet_bayes
from collections import OrderedDict

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (0.6 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm



def fit(X, num_classes, percentage, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind

def data_split(data, clean_labels, noisy_labels, split_percentage, seed=1):
   
    num_samples = int(clean_labels.shape[0])
    np.random.seed(int(seed))
    train_set_index = np.random.choice(num_samples, int(num_samples*split_percentage), replace=False)
    index = np.arange(data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = data[train_set_index, :], data[val_set_index, :]
    train_clean_labels, val_clean_labels = clean_labels[train_set_index], clean_labels[val_set_index]
    train_noisy_labels, val_noisy_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_noisy_labels, val_noisy_labels,train_clean_labels, val_clean_labels,


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target    

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-1)
            
    return net



def distill_examples(rho, model_dir, classifier,train_loader_batch_1,txtfile,train_dataset,batch_size,dataset='cifar10'):
    # Distlled example collection
    threshold = (1 + rho) / 2 
    distilled_example_index_list = []
    distilled_example_labels_list = []
    print('Distilling')
    classifier.eval()
    for i, (data, noisy_label, clean_label, indexes) in enumerate(tqdm(train_loader_batch_1)):
        data = data.cuda()
        logits1= F.softmax(classifier(data), dim=1)
        if torch.max(logits1) > threshold:
            distilled_example_index_list.append(indexes.item())
            distilled_example_labels_list.append(int(torch.argmax(logits1).cpu().numpy()))
    print("Distilling finished")
    distilled_example_index = np.array(distilled_example_index_list)
    distilled_bayes_labels = np.array(distilled_example_labels_list)
    distilled_images, distilled_noisy_labels, distilled_clean_labels  = train_dataset.train_data[distilled_example_index], train_dataset.train_noisy_labels[distilled_example_index],train_dataset.train_clean_labels[distilled_example_index] # noisy labels
    print("Number of distilled examples:"+str(len(distilled_bayes_labels)))
    print("Accuracy of distilled examples collection:"+ str((np.array(distilled_bayes_labels) ==  np.array(distilled_clean_labels)).sum() / len(distilled_bayes_labels)))
    with open(txtfile, "a") as myfile:
        myfile.write("Number of distilled examples:"+str(len(distilled_bayes_labels))+'\n')
        myfile.write("Accuracy of distilled examples collection:"+ str((np.array(distilled_bayes_labels) ==  np.array(distilled_clean_labels)).sum() / len(distilled_bayes_labels)) + '\n')
    np.save(model_dir+'/'+'distilled_images.npy',distilled_images)
    np.save(model_dir+'/'+'distilled_bayes_labels.npy',distilled_bayes_labels)
    np.save(model_dir+'/'+'distilled_noisy_labels.npy',distilled_noisy_labels)
    np.save(model_dir+'/'+'distilled_clean_labels.npy',distilled_clean_labels)
    
    print("Distilled dataset building")
    import data
    distilled_images = np.load(model_dir+'/'+'distilled_images.npy')
    distilled_noisy_labels = np.load(model_dir+'/'+'distilled_noisy_labels.npy')
    distilled_bayes_labels = np.load(model_dir+'/'+'distilled_bayes_labels.npy')
    distilled_clean_labels = np.load(model_dir+'/'+'distilled_clean_labels.npy')
        
    if dataset =='mnist':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target
                                        )
    if dataset == 'cifar10':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),]),
                                        target_transform=transform_target
                                        ) 
    if dataset == 'svhn':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),]),
                                        target_transform=tools.transform_target
                                        )
        
    
    train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               drop_last=False,
                                               shuffle=False)
    return train_loader_distilled