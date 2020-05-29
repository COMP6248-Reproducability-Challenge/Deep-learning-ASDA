##evaluate


#CIFAR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import json
import time
import sys
import copy


from kWTA import models
from kWTA import activation
from kWTA import attack
from kWTA import training
from kWTA import utilities
from kWTA import densenet
from kWTA import resnet
from kWTA import wideresnet

norm_mean = 0
norm_var = 1

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])

cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)


## natural


print("------------nature-----------")
device = torch.device('cuda:0')
model = resnet.ResNet18().to(device)
print("model loading --------Relu")
model.load_state_dict(torch.load('models/resnet18_cifar_80epochs.pth'))
eps = 0.031
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)
  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 
test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.1,0.1,0.1,0.1], sparse_func='vol').to(device)
print("model loading --------k=0.1")
model.load_state_dict(torch.load('models/spresnet18_0.1_cifar_80epochs.pth'))


test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)  


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.2,0.2,0.2,0.2], sparse_func='vol').to(device)
print("model loading --------k=0.2")
model.load_state_dict(torch.load('models/spresnet18_0.2_cifar_80epochs.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err) 



## AT


print("------------AT-----------")
device = torch.device('cuda:0')
model = resnet.ResNet18().to(device)
print("model loading --------Relu")
model.load_state_dict(torch.load('models/resnet18_cifar_adv_80epochs.pth'))
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)
  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 
test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.1,0.1,0.1,0.1], sparse_func='vol').to(device)
print("model loading --------k=0.1")
model.load_state_dict(torch.load('models/spresnet18_0.1_cifar_adv.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)  


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.2,0.2,0.2,0.2], sparse_func='vol').to(device)
print("model loading --------k=0.2")

model.load_state_dict(torch.load('models/spresnet18_0.2_cifar_adv.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)







## FAT


print("------------FAT-----------")
device = torch.device('cuda:0')
model = resnet.ResNet18().to(device)
print("model loading --------Relu")
model.load_state_dict(torch.load('models/resnet18_cifar_free_83epochs.pth'))
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)
  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 
test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.1,0.1,0.1,0.1], sparse_func='vol').to(device)
print("model loading --------k=0.1")
model.load_state_dict(torch.load('models/FAT_resnet18_0.1_cifar.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)  


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.2,0.2,0.2,0.2], sparse_func='vol').to(device)
print("model loading --------k=0.2")
model.load_state_dict(torch.load('models/spresnet18_0.2_cifar_free_80.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)  



## TRADE


print("------------TRADE-----------")
device = torch.device('cuda:0')
model = resnet.ResNet18().to(device)
print("model loading --------Relu")
model.load_state_dict(torch.load('models/resnet18_cifar_trades_epoch80.pth'))
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)
  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 
test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.1,0.1,0.1,0.1], sparse_func='vol').to(device)
print("model loading --------k=0.1")
model.load_state_dict(torch.load('models/resnet18_0.1_cifar_trades_epoch79.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err)  


device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[0.2,0.2,0.2,0.2], sparse_func='vol').to(device)
print("model loading --------k=0.2")
model.load_state_dict(torch.load('models/spresnet18_cifar_k02_trades_epoch79.pth'))

test_loader = DataLoader(cifar_test, batch_size = 400, shuffle=True)
model.eval()
test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
print("test",test_err)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003)
print("PGD:",adv_err)


adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.CW, device=device, num_iter=20, 
        use_tqdm=True)
print("CW:",adv_err)  

adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.MIM, device=device, num_iter=20, 
        use_tqdm=True)
print("MIM:",adv_err) 

test_loader = DataLoader(cifar_test, batch_size = 1, shuffle=True)
adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.deepfool, device=device, num_iter=20, 
        use_tqdm=True, epsilon=eps,n_test=1000)
print("Deepfool:",adv_err) 

