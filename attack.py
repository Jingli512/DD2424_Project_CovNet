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
import numpy as np
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
    
    # model.train()
    return x_adv

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--model', default = 'vanilla', help='training batch size')
parser.add_argument('--attack_iter', type=int, default = 10, help='training batch size')
parser.add_argument('--epsilon', type=float, default = 8.0 / 255.0, help='Learning Rate')
parser.add_argument('--batch_norm', default=False, action = 'store_true', help='save model path')
parser.add_argument('--dropout', default=None, type = float, action = 'append', help='Should be a list. Usage: python train.py --dropout 0.2 --dropout 0.3 --dropout 0.4. Then args.dropout will be: [0.2, 0.3, 0.4]')
args = parser.parse_args()
print('Attack: ', args.attack_iter, int(args.epsilon * 255.0))
'''Different Data Loading Schemes'''
transform_test = transforms.Compose([
        transforms.ToTensor()
])
'''Loading Data'''
test_data = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        transform = transform_test,
        download = True
)
test_loader = Data.DataLoader(dataset = test_data, batch_size = 100, shuffle = False,num_workers=2)
'''Create Model'''
if(use_cuda):
    model = VGG_3block(batch_norm = args.batch_norm, dropout = args.dropout).cuda()
else:
    model = VGG_3block(batch_norm = args.batch_norm, dropout = args.dropout)

model.load_state_dict(torch.load(f'./model/{args.model}.pkl'))

'''Define Loss'''
loss_func = nn.CrossEntropyLoss()

total = 0
correct = 0
correct_adv = 0
test_loss = 0
total_test_step = 0
model.eval()
for test_step,(val_x,val_y) in tqdm(enumerate(test_loader)):
    val_x = val_x.cuda()
    val_y = val_y.cuda()

    adv_x = PGD_attack(model = model, epsilon = args.epsilon, k = args.attack_iter, a = args.epsilon / args.attack_iter, x_nat = val_x, y = val_y, loss_func = loss_func)

    val_output = model(val_x)
    loss = loss_func(val_output, val_y)

    val_output_adv = model(adv_x)

    test_loss += loss.item()
    total_test_step += 1

    _, val_pred_y = val_output.max(1)
    correct += val_pred_y.eq(val_y).sum().item()

    _, val_pred_y_adv = val_output_adv.max(1)
    correct_adv += val_pred_y_adv.eq(val_y).sum().item()

    if(test_step == 0):
        idx = 10
        img = np.array(transforms.ToPILImage()(val_x[idx].cpu()))
        plt.imshow(img)
        plt.title(f'Original, Pred: {int(val_pred_y[idx].cpu().detach())}, True: {int(val_y[idx].cpu().detach())}', fontsize = 17)
        plt.axis('off')
        plt.savefig('image/ori.png')
        plt.clf()

        img = np.array(transforms.ToPILImage()(adv_x[idx].cpu()))
        plt.imshow(img)
        plt.title(f'Eps {int(args.epsilon * 255.0)}, iteration {args.attack_iter}, Pred: {int(val_pred_y_adv[idx].cpu().detach())}, True: {int(val_y[idx].cpu().detach())}', fontsize = 17)
        plt.axis('off')
        plt.savefig(f'image/adv_{int(args.epsilon * 255.0)}_{args.attack_iter}.png')
        plt.clf()

    total += val_y.size(0)

# test_loss_mean = test_loss / total_test_step
# test_losses.append(test_loss_mean)
# test_acc = correct / total
# test_accs.append(test_acc)

result = float(correct) * 100.0 / float(total)
result_adv = float(correct_adv) * 100.0 / float(total)

print(f'Test Accuracy: {result}, Adv Accuracy: {result_adv}')  
