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
import math

def PGD_attack(model,epsilon,k,a,x_nat,y,loss_func):
    model.eval()
    x_rand = x_nat.detach()
    x_rand = x_rand + torch.zeros_like(x_rand).uniform_(-epsilon,epsilon)
    x_adv = Variable(x_rand.data, requires_grad = True).cuda()
    
    for j in range(k):
        h_adv = model(x_adv)
        loss = loss_func(h_adv, y)
        model.zero_grad()
        if(x_adv.grad is not None):
            x_adv.grad.data.fill_(0)
        loss.backward()
        
        x_adv = x_adv.detach() + a * torch.sign(x_adv.grad)
        x_adv = torch.where(x_adv > x_nat+epsilon, x_nat+epsilon, x_adv)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = torch.where(x_adv < x_nat-epsilon, x_nat-epsilon, x_adv)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = Variable(x_adv.data, requires_grad=True).cuda()
    
    model.train()
    return x_adv

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--custom_data', default=False, action = 'store_true', help='Whether to use customized Dataset')
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
parser.add_argument('--record_train', default=False, action = 'store_true', help='save model path')
parser.add_argument('--attack_iter', type=int, default = 10, help='training batch size')
parser.add_argument('--epsilon', type=float, default = 8.0 / 255.0, help='Learning Rate')

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
if(args.custom_data == True):
    with open('./train/train_data.pkl', 'rb') as f:
        imgs = pickle.load(f)
    with open('./train/train_label.pkl', 'rb') as f:
        labels = pickle.load(f)
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
train_loader = Data.DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True,num_workers=2)
test_loader = Data.DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False,num_workers=2)
'''Create Model'''
if(use_cuda):
    model = VGG_3block(batch_norm = args.batch_norm, dropout = args.dropout).cuda()
else:
    model = VGG_3block(batch_norm = args.batch_norm, dropout = args.dropout)

'''Create Optimizer'''
if(args.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum,weight_decay = args.weight_decay)
elif(args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

'''Create LR scheduling, such as cosine, etc.'''
if(args.lr_scheduler is not None):
    eta_min = args.lr * 0.1
    eta_max = args.lr * 10

'''Define Loss'''
loss_func = nn.CrossEntropyLoss()

'''Start Training Process'''
model.train()

train_losses = []
test_losses = []
train_accs = []
test_accs = []
lrs = []
epoch_warmup = 5
for epoch in tqdm(range(args.epochs)):
    train_loss = 0
    total_step = 0
    train_correct = 0
    train_total_data = 0
    if(args.lr_scheduler is not None):
        if(args.lr_scheduler == 'cosine'):
            if(epoch < epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_min + (eta_max - eta_min) * epoch / (epoch_warmup)
            if(epoch == epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_max
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min = eta_min, T_max=15)
            if(epoch > epoch_warmup):
                lr_scheduler.step()
        elif(args.lr_scheduler == 'step'):
            if(epoch < epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_min + (eta_max - eta_min) * epoch / (epoch_warmup)
            if(epoch == epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_max
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 3, args.epochs // 3 * 2], gamma=0.1)
            if(epoch > epoch_warmup):
                lr_scheduler.step()
        elif(args.lr_scheduler == 'cosine_restart'):
            if(epoch < epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_min + (eta_max - eta_min) * epoch / (epoch_warmup)
            if(epoch == epoch_warmup):
                optimizer.param_groups[0]['lr'] = eta_max
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 15, eta_min = eta_min)
            if(epoch > epoch_warmup):
                lr_scheduler.step() 
        lrs.append(optimizer.param_groups[0]['lr'])
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()

        logits = model(x)
        loss_vanilla = loss_func(logits, y)

        if(epoch >= 50):
            adv_x = PGD_attack(model = model, epsilon = args.epsilon, k = args.attack_iter, a = args.epsilon / args.attack_iter, x_nat = x, y = y, loss_func = loss_func)
            logits_adv = model(adv_x)
            loss_adv = loss_func(logits_adv, y)
        
            loss = loss_vanilla + loss_adv
        else:
            loss = loss_vanilla
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if(args.record_train):
            train_loss += loss.item()
            total_step += 1

            _,train_pred_y = logits.max(1)
            train_correct += train_pred_y.eq(y).sum().item()
            train_total_data += y.size(0)
    if(args.record_train):
        train_loss_mean = train_loss / total_step
        train_losses.append(train_loss_mean)
        train_acc = train_correct / train_total_data
        train_accs.append(train_acc)
        if(args.not_print_process == False):
            print(f'Epoch {epoch}, Loss {train_loss_mean}, Acc {train_acc}')

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
            loss = loss_func(val_output, val_y)

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
    model.train()    

torch.save(model.state_dict(), f'./model/pat_{int(args.epsilon * 255.0)}_{args.attack_iter}.pkl')

if(args.lr_scheduler is not None):
    plt.plot(lrs)
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Learning rate', fontsize = 15)
    plt.savefig(f'image/lr_{args.lr_scheduler}.png')
    plt.clf()

plt.plot(train_losses, label = 'Train Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.legend(fontsize = 15)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.savefig(f'image/loss_pat_{int(args.epsilon * 255.0)}_{args.attack_iter}.png')
plt.clf()

plt.plot(train_accs, label = 'Train Accuracy')
plt.plot(test_accs, label = 'Test Accuracy')
plt.legend(fontsize = 15)
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.savefig(f'image/acc_pat_{int(args.epsilon * 255.0)}_{args.attack_iter}.png')
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
