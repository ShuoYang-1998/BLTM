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
import tools
import resnet
import resnet_bayes
from collections import OrderedDict



parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5, help="No.")
parser.add_argument('--d', type=str, default='Bayesian-T_cifar10', help="description")
parser.add_argument('--p', type=int, default=1, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results_ours')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.3)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='instance')
parser.add_argument('--dataset', type=str, help='fmnist, cifar10,svhn', default='svhn')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='ours')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--nonzero_ratio', type=float, help='choose pruning ratio', default=0.2)
parser.add_argument('--split_per', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
parser.add_argument('--split_percentage', type = float, help = 'train and validation', default=0.9)
parser.add_argument('--rho', type = float, help = 'rho', default=0.1)

args = parser.parse_args()
#
torch.cuda.set_device(args.gpu)

print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.dataset=='fmnist':
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 20
        train_dataset = data.fashionmnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)

        val_dataset = data.fashionmnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)


        test_dataset =  data.fashionmnist_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)

    if args.dataset=='cifar10':
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 20
        train_dataset = data.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)

        val_dataset = data.cifar10_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)


        test_dataset =  data.cifar10_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target)

    if args.dataset=='svhn':
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 20
        train_dataset = data.svhn_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)

        val_dataset = data.svhn_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        noise_rate=args.noise_rate,
                                        split_percentage=args.split_percentage,
                                        seed=args.seed)


        test_dataset =  data.svhn_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                        ]),
                                        target_transform=tools.transform_target)

    return train_dataset, val_dataset, test_dataset



save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % str(args.noise_rate)+'_'+str(args.rho)




def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model


def train_one_step(net, data, label, optimizer, criterion):
    net.train()
    pred = net(data)
    loss = criterion(pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy(pred, label, topk=(1,))

    return float(acc[0]), loss


def train(train_loader, epoch, model1, optimizer1):
    print('Training %s...' % model_str)
    model1.train()
    train_total = 0
    train_correct = 0
    
    for i, (data, noisy_label, clean_label, indexes) in enumerate(train_loader):
        
        ind = indexes.cpu().numpy().transpose()
        data = data.cuda()
        labels = noisy_label.cuda()

        # Forward + Backward + Optimize
        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        # Loss transfer

        # prec1, loss = train_one_step(model, data, labels, optimizer1, nn.CrossEntropyLoss(), 1-args.noise_rate, clip)
        prec1, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss())

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                  % (epoch + 1, i + 1, 50000 // batch_size, prec1, loss.item()))

    train_acc1 = float(train_correct) / float(train_total)

    return train_acc1


# Evaluate the Model
def evaluate(val_loader, model1):
    print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, noisy_label, clean_label, _ in val_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += noisy_label.size(0)
            correct1 += (pred1.cpu() == clean_label.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def train_forward(model, train_loader, epoch, optimizer, Bayesian_T, revision=True):
    
    train_total=0
    train_correct=0 

    for i, (data, labels, _, indexes) in enumerate(train_loader):
        
        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        logits, delta = model(data, revision=True)
        bayes_post = F.softmax(logits, dim=1)
        
        delta = delta.repeat(len(labels),1,1)
        T = Bayesian_T(data)
        if revision == True:
            T = tools.norm(T + delta)
        noisy_post = torch.bmm(bayes_post.unsqueeze(1),T.cuda()).squeeze(1)
        log_noisy_post = torch.log(noisy_post+1e-12)
        loss = nn.NLLLoss()(log_noisy_post.cuda(),labels.cuda())
        
        prec1,  = accuracy(noisy_post, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        loss.backward()
        optimizer.step()
        
    train_acc=float(train_correct)/float(train_total)
    return train_acc


def main(args):
#     # Data Loader (Input Pipeline)

    model_dir = save_dir + str(args.seed)+'_rate_'+str(args.noise_rate)+'_rho_'+str(args.rho)
    if not os.path.exists(model_dir):
        os.system('mkdir -p %s' % model_dir)
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)
    
    train_loader_batch_1 = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    
#     # Define models

    print('building model...')
    if args.dataset == 'fmnist':
        classifier = resnet.ResNet18_F(10).cuda()
    if args.dataset == 'svhn':
        classifier = resnet.ResNet34(10).cuda()
    if args.dataset == 'cifar10':
        classifier = resnet.ResNet34(10).cuda()


    classifier.cuda()
    cudnn.benchmark = True


    # Warm up classifier to distill examples
    val_acc_list = []
    test_acc_list = []
    best_acc = 0.
    classifier.cuda()
    cudnn.benchmark = True
    optimizer_warmup = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    print('starting warm up')
    for epoch in range(0, 10):
        classifier.train()
        train_acc1 = train(train_loader, epoch, classifier, optimizer_warmup)
        val_acc1 = evaluate(test_loader, classifier)
        print('Warm up Epoch [%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
        epoch + 1, len(test_dataset), val_acc1))
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(classifier.state_dict(), model_dir + '/' + 'warmup_model.pth')
    
    
    # Distlled example collection
    threshold = (1 + args.rho) / 2 
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    test_acc1 = evaluate(test_loader, classifier)
    print('Loading Test Accuracy on the %s val data: Model1 %.4f %%' % (
         len(test_dataset), test_acc1))

    distilled_example_index_list = []
    distilled_example_labels_list = []
    print('Distilling')
    classifier.eval()
    for i, (data, noisy_label, clean_label, indexes) in enumerate((train_loader_batch_1)):
        data = data.cuda()
        logits1= F.softmax(classifier(data), dim=1)
        logits1_max = torch.max(logits1,dim=1)
        mask = logits1_max[0]>threshold
        distilled_example_index_list.extend(indexes[mask])
        distilled_example_labels_list.extend(logits1_max[1].cpu()[mask])
    print("Distilling finished")
    distilled_example_index = np.array(distilled_example_index_list)
    distilled_bayes_labels = np.array(distilled_example_labels_list)
    distilled_images, distilled_noisy_labels, distilled_clean_labels  = train_dataset.train_data[distilled_example_index], train_dataset.train_noisy_labels[distilled_example_index],train_dataset.train_clean_labels[distilled_example_index] # noisy labels
    print("Number of distilled examples:"+str(len(distilled_bayes_labels)))
    print("Accuracy of distilled examples collection:"+ str((np.array(distilled_bayes_labels) ==  np.array(distilled_clean_labels)).sum() / len(distilled_bayes_labels)))
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
        
    if args.dataset =='fmnist':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target
                                        )
    if args.dataset == 'cifar10':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),]),
                                        target_transform=tools.transform_target
                                        ) 
    if args.dataset == 'svhn':
    
        distilled_dataset_= data.distilled_dataset(distilled_images,
                                        distilled_noisy_labels,
                                        distilled_bayes_labels,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),]),
                                        target_transform=tools.transform_target
                                        )
        
    
    train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True)

    if args.dataset == 'fmnist':
        Bayesian_T_Network = resnet_bayes.ResNet18_F(100)
        warm_up_dict = classifier.state_dict()
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    if args.dataset == 'svhn':
        Bayesian_T_Network = resnet_bayes.ResNet34(100)
        warm_up_dict = classifier.state_dict()
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    if args.dataset == 'cifar10':
        Bayesian_T_Network = resnet_bayes.ResNet34(100)
        warm_up_dict = classifier.state_dict()
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
#         for name, parameter in Bayesian_T_Network.named_parameters():
#             if 'bayes_linear' not in name:
#                 parameter.requires_grad = False
    Bayesian_T_Network.cuda()
    #Learning Bayes T
#     clf_bayes_output -> transition matrix with size c*c
    optimizer_bayes = torch.optim.SGD(Bayesian_T_Network.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.NLLLoss()
    for epoch in range(0, 50):
        bayes_loss = 0.
        Bayesian_T_Network.train()
        for data, bayes_labels, noisy_labels, index in train_loader_distilled:
            data = data.cuda()
            bayes_labels, noisy_labels = bayes_labels.cuda(), noisy_labels.cuda()
            # Forward + Backward + Optimize
            batch_matrix = Bayesian_T_Network(data)# batch_size x 10 x 10
            noisy_class_post = torch.zeros((batch_matrix.shape[0], 10))
            for j in range(batch_matrix.shape[0]):
                bayes_label_one_hot = torch.nn.functional.one_hot(bayes_labels[j], 10).float() # 1*10
                bayes_label_one_hot = bayes_label_one_hot.unsqueeze(0)
                noisy_class_post_temp = bayes_label_one_hot.float().mm(batch_matrix[j]) # 1*10 noisy
                noisy_class_post[j, :] = noisy_class_post_temp
        noisy_class_post = torch.log(noisy_class_post+1e-12)
        loss = loss_function(noisy_class_post.cuda(), noisy_labels)
        optimizer_bayes.zero_grad()
        loss.backward()
        optimizer_bayes.step()
        bayes_loss += loss.item()
        print('Bayesian-T Training Epoch [%d], Loss: %.4f'% (epoch + 1, loss.item()))
        torch.save(Bayesian_T_Network.state_dict(), model_dir + '/' + 'BayesianT.pth')


    # loss_correction
    val_acc_list = []
    test_acc_list = []

    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    nn.init.constant_(classifier.T_revision.weight, 0.0)
    
    Bayesian_T_Network.load_state_dict(torch.load(model_dir + '/' + 'BayesianT.pth'))
    print('Loading Test Accuracy on the %s test data: Model1 %.4f %%' % (
        len(test_dataset), evaluate(test_loader, classifier)))
    optimizer_r = torch.optim.Adam(classifier.parameters(), lr=5e-7, weight_decay=1e-4)
    
    for epoch in range(0, args.n_epoch):
        classifier.train()
        Bayesian_T_Network.eval()
        train_total = 0
        train_correct = 0
        train_acc = train_forward(classifier,train_loader,epoch,optimizer_r,Bayesian_T_Network,revision=True)
        test_acc = evaluate(test_loader, classifier)
        test_acc_list.append(test_acc)
        # save results
        print('Epoch [%d/%d] Train Accuracy on the %s train data: Model1 %.4f %% ' % (
            epoch + 1, args.n_epoch, len(train_dataset), train_acc))
#         print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %% ' % (
#             epoch + 1, args.n_epoch, len(val_dataset), val_acc1))
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc))
        
    id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[id]
    print('Test Acc: ')
    print(test_acc_max)
    return test_acc_max


    
            
if __name__ == '__main__':
    acclist = []
    for i in range(args.n):
        args.seed = i + 1
        args.output_dir = './' + args.d + '/' + str(args.noise_rate) + '/'
        if not os.path.exists(args.output_dir):
            os.system('mkdir -p %s' % (args.output_dir))
        if args.p == 0:
            f = open(args.output_dir + str(args.noise_type) + '_' + str(args.dataset) + '_' + str(args.rho) + '.txt', 'a')
            sys.stdout = f
            sys.stderr = f
        acc = main(args)
        acclist.append(acc)
    print(np.array(acclist).mean())
    print(np.array(acclist).std(ddof=1))