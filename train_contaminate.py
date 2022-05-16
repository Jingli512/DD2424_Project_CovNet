import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import pickle
from VGG import VGG_3block
from Dataset import Custom_CIFAR10
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np

'''
def one_hot(a):
    a = np.array(a)
    result = np.zeros((a.size, a.max()+1))
    result[np.arange(a.size),a] = 1
    return result

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#y_true: (n,)
#y_pred: (n, 10)
def symmetric_cross_entropy2(alpha, beta, y_pred, y_true):
    y_true = one_hot(y_true)
    y_pred = softmax(y_pred)
    ce = -alpha * (1/y_pred.shape[0]) * np.sum(y_true * np.log(y_pred))
    rce = -beta * (1/y_pred.shape[0]) * np.sum(y_pred * np.log(y_true))
    return ce+rce
'''

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--custom_data', default=True, action = 'store_true', help='Whether to use customized Dataset')
parser.add_argument('--noise_level',type=float, default=0, help='Noise Level')
parser.add_argument('--lossType', default='CE', help='Loss Type')
parser.add_argument('--optimizer', default="SGD", help='load model path')
parser.add_argument('--lr_scheduler', default=None, help='load model path')
parser.add_argument('--normalize_data', default=False, action = 'store_true', help='load model path')
parser.add_argument('--data_aug', default=False, action = 'store_true', help='save model path')
parser.add_argument('--batch_norm', default=False, action = 'store_true', help='save model path')
parser.add_argument('--dropout', default=None, type = float, action = 'append', help='Should be a list. Usage: python train.py --dropout 0.2 --dropout 0.3 --dropout 0.4. Then args.dropout will be: [0.2, 0.3, 0.4]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=0, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--not_print_process', default=False, action = 'store_true', help='save model path')

args = parser.parse_args()
'''Different Data Loading Schemes'''
if(args.normalize_data == True):
    if(args.data_aug == True):
        print('Data Aug with normalization')
        transform_train = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate = (0.1, 0.1)),
                transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.ToTensor()
    ])
elif(args.normalize_data == False):
    if(args.data_aug == True):
        print('Data Aug with no normalization')
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate = (0.1, 0.1)),
                transforms.ToTensor()
        ])
    else:
        transform_train = transforms.Compose([
                transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor()
    ])
'''Loading Data'''
#noise_level = 0.01
if(args.custom_data == True):
    with open('./train/train_data.pkl', 'rb') as f:
        imgs = pickle.load(f)
    with open('./train/train_label.pkl', 'rb') as f:
        labels = pickle.load(f)
        change_idx_dic = {}
        for i in range(10):
            class_idx = torch.where(labels==i)[0].numpy()
            #replace=False: the same index cannot be picked several times
            change_idx_dic[i] = np.random.choice(class_idx, size=int(len(class_idx)*args.noise_level), replace=False)
        for i in range(10):   
            labels[change_idx_dic[i]] = (i+1)%10
    train_data = Custom_CIFAR10(imgs = imgs, labels = labels, transform = transform_train)
else:
    train_data = torchvision.datasets.CIFAR10(
            root = './data',
            train = True,
            transform = transform_train,
            download = True
    )
test_data = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        transform = transform_test,
        download = True
)
train_loader = Data.DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True, num_workers=2)
test_loader = Data.DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False, num_workers=2)
'''Create Model'''
model = VGG_3block(batch_norm = args.batch_norm, dropout = args.dropout).cuda()

'''Create Optimizer'''
if(args.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum,weight_decay = args.weight_decay)
elif(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

'''Create LR scheduling, such as cosine, etc.'''
if(args.lr_scheduler is not None):
    pass

'''Define Loss'''
#loss_type_CE = True
if (args.lossType == 'CE'):
    loss_func = nn.CrossEntropyLoss()
if (args.lossType == 'SL'):
    alpha = 0.1
    beta = 1.0
    SCE = SCELoss(alpha, beta)

'''Start Training Process'''
model.train()

train_losses = []
test_losses = []
train_accs = []
test_accs = []
for epoch in tqdm(range(args.epochs)):
    train_loss = 0
    total_step = 0
    train_correct = 0
    train_total_data = 0
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()

        logits = model(x)

        #print("y",y.cpu().numpy())
        #print("logits", logits.detach().cpu().numpy())
        if (args.lossType == 'CE'):
            loss = loss_func(logits, y)
        #loss = symmetric_cross_entropy2(1.0, 1.0, logits.detach().cpu().numpy(), y.cpu().numpy())
        if (args.lossType == 'SL'):
            loss = SCE.forward(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_step += 1

        _,train_pred_y = logits.max(1)
        train_correct += train_pred_y.eq(y).sum().item()
        train_total_data += y.size(0)

    train_loss_mean = train_loss / total_step
    train_losses.append(train_loss_mean)
    train_acc = train_correct / train_total_data
    train_accs.append(train_acc)
    #if(args.not_print_process == False):
    #    print(f'Epoch {epoch}, Loss {train_loss_mean}, Acc {train_acc}')

    total = 0
    correct = 0
    test_loss = 0
    total_test_step = 0
    model.eval()
    with torch.no_grad():
        for test_step,(val_x,val_y) in enumerate(test_loader):
            val_x = val_x.cuda()
            val_y = val_y.cuda()
            val_output = model(val_x)
            if (args.lossType == 'CE'):
                loss = loss_func(val_output, val_y)
            if (args.lossType == 'SL'):
                loss = SCE.forward(val_output, val_y)

            test_loss += loss.item()
            total_test_step += 1

            _, val_pred_y = val_output.max(1)
            correct += val_pred_y.eq(val_y).sum().item()
            total += val_y.size(0)

    test_loss_mean = test_loss / total_test_step
    test_losses.append(test_loss_mean)
    test_acc = correct / total
    test_accs.append(test_acc)

    result = float(correct) * 100.0 / float(total)
    if(args.not_print_process == False):
        print(f'Test Accuracy: {result}')
        print("Current mean test loss: ", test_loss_mean)
    model.train()

plt.plot(train_losses, label = 'Train Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.legend(fontsize = 15)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.savefig(f'image/loss_{args.optimizer}_{args.lr_scheduler}_{args.normalize_data}_{args.data_aug}_{args.dropout}_{args.weight_decay}.png')
plt.clf()

plt.plot(train_accs, label = 'Train Accuracy')
plt.plot(test_accs, label = 'Test Accuracy')
plt.legend(fontsize = 15)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.savefig(f'image/acc_{args.optimizer}_{args.lr_scheduler}_{args.normalize_data}_{args.data_aug}_{args.dropout}_{args.weight_decay}.png')
plt.clf()

print('Final Test')

total = 0
correct = 0
model.eval()
with torch.no_grad():
    for test_step,(val_x,val_y) in enumerate(test_loader):
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        _,val_pred_y = val_output.max(1)
        correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
result = float(correct) * 100.0 / float(total)
print('Test Accuracy: %.2f%%' % result)
