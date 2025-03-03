#add todo list
#add another transofmer and predict all layer. 
#think if all central feautres are to be added with mean or all feautres as an input. 










#done below
#add ALL BN prediction and updation 
#add tent based entropy minimization in starting part of meta adapt loop 
#from tent take the gradients 

#mimic domain shifts 



# train on different ditributions in meta train and meta test 
# add transfomer seperately and optimize it wirh Cross entropy of model created and model old
# capture Bn statistics and log to wandb
#upload these stats to github 
#add gradients list from SF as a matrix to transformer. 
#learning rates just be mindful 
#remove SF for now in this version 

from __future__ import print_function
#do test after every epoch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import os, shutil
import argparse
import time
#import tensorboard from torch 
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from aug import *
import pdb
from pacs_rtdataset_new import *
from pacs_dataset import *
# import dg_model_tr_net
# import pacs_model
# import dg_meta_model
# import dg_model
import dg_model_tr_net_layer67
import sys
import numpy as np
from torch.nn import init
from sklearn.model_selection import train_test_split
import wandb
bird = False
wandb.init(project="server_project_grad_train")
import copy 

from math import remainder
#meta imports 
import transformer_model_confg_67 
import dg_model_layer67

import math 

import pdb
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--dataset', default='PACS', help='datasets')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 32]')
parser.add_argument('--pseudo_label_update_epoch', default=10, type=int, help='epoch to update pseudo labels')
# parser.add_argument('--bases', type=int, default=7, help='Batch Size during training [default: 32]')
parser.add_argument('--shuffle', type=int, default=0, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='AdamW', help='AdamW or momentum [default: AdamW]')
# parser.add_argument('--sharing', default='layer', help='Log dir [default: log]')
parser.add_argument('--net', default='res18', help='res18 or res50')
# parser.add_argument('--l2', action='store_true')
# parser.add_argument('--base', action='store_true')
parser.add_argument('--autodecay', action='store_true')
# parser.add_argument('--share_bases', action='store_true')
# parser.add_argument('--hychy', type=int, default=0, help='hyrarchi')
# parser.add_argument('--sub', default=1.0, type=float, help='subset of tinyimagenet')
parser.add_argument('--test_domain', default='art_painting', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_domain', default='', help='GPU to use [default: GPU 0]')
parser.add_argument('--ite_train', default=True, type=bool, help='learning rate')
parser.add_argument('--max_ite', default=10000, type=int, help='max_ite')
parser.add_argument('--test_ite', default=50, type=int, help='learning rate')
parser.add_argument('--bias', default=1, type=int, help='whether sample')
parser.add_argument('--test_batch', default=100, type=int, help='learning rate')
parser.add_argument('--data_aug', default=1, type=int, help='whether sample')
parser.add_argument('--difflr', default=1, type=int, help='whether sample')
# parser.add_argument('--mc_times', default=10, type=int, help='number of Monte Carlo samples')
# parser.add_argument('--mbeta', default=1e-3, type=float, help='beta for mid y')
# parser.add_argument('--abeta', default=1e-5, type=float, help='beta for adaptive kl')
# parser.add_argument('--alpha', default=0.5, type=float, help='beta for adaptive kl')
# parser.add_argument('--norm', default='bn', help='bn or in')
# parser.add_argument('--domain_l', default=4, type=int, help='1 or 3 or 4')
# parser.add_argument('--test_sample', default=False, type=bool, help='sampling in test time')
# parser.add_argument('--dinit', default='rt', help='random r or feature f or center feature c')
parser.add_argument('--reslr', default=0.5, type=float, help='backbone learning rate')
# parser.add_argument('--pbeta', default=1, type=float, help='backbone learning rate')
parser.add_argument('--agg_model', default='concat', help='concat or bayes or rank1')
parser.add_argument('--agg_method', default='mean', help='ensemble or mean or ronly')
# parser.add_argument('--dom_sta', default='mean', help='both or mean')
# parser.add_argument('--ptest', default=0, type=int, help='use prior in test')
# parser.add_argument('--sharem', default=0, type=int, help='share model or not')
parser.add_argument('--ctx_num', default=10, type=int, help='learning rate')
parser.add_argument('--hierar', default=2, type=int, help='hierarchical model')
# parser.add_argument('--adp_num', default='1', help='learning rate')
# parser.add_argument('--mul_tra_sam', default=0, type=int, help='hierarchical model')
# parser.add_argument('--dsgrad', default=1, type=int, help='whether sample')
parser.add_argument('--model_saving_dir', default= './models_new/models_code', type = str, help=' place to save the best model obtained during training')
#add resume from checkpoint
parser.add_argument('--resume_from_checkpoint', type = str, help=' resume from checkpoint')
#number of tr_net layers 
parser.add_argument('--tr_net_layers', type = int, default = 8, help='number of tr_net layers')
#add resnet learning rate 
args = parser.parse_args()
wandb.config.update(args) 

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net
# dsgrad = bool(args.dsgrad)
max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain
# num_adp = args.adp_num
# mid_beta = args.mbeta
ctx_num = args.ctx_num
# a_beta = args.abeta
# multi_train_sam = bool(args.mul_tra_sam)
# p_beta = args.pbeta
# ifsample = args.test_sample
# Dinit = args.dinit
# dom_sta = args.dom_sta
difflr = args.difflr
res_lr = args.reslr
hierar = args.hierar
agg_model = args.agg_model
# agg_method = args.agg_method
# prior_test = args.ptest
# prior_test = bool(prior_test)
with_bias = args.bias
with_bias = bool(with_bias)
difflr = bool(difflr)
pseudo_label_update_epoch = args.pseudo_label_update_epoch
# sharemodel = bool(args.sharem)
# alpha = args.alpha

# norm_method = args.norm
# zdlayers = args.domain_l

# mc_times = args.mc_times
# ifcommon = args.ifcommon
# ifadapt = args.ifadapt

data_aug = args.data_aug
data_aug = bool(data_aug)
model_saving_dir = args.model_saving_dir
resume_from_checkpoint = args.resume_from_checkpoint
tr_net_layers = args.tr_net_layers


#LOG_DIR = os.path.join('logs', args.log_dir)
wandb_run_name = wandb.run.name
# torch.cuda.is_available()
# exit()
LOG_DIR = os.path.join('logs_meta', wandb_run_name)
MODEL_DIR = os.path.join(model_saving_dir, wandb_run_name)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)



#tensorboard start
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
#create a validation directoryt to store the validation results
if not os.path.exists(os.path.join(LOG_DIR, 'validation')):
    os.makedirs(os.path.join(LOG_DIR, 'validation'))
#create a test directoryt to store the test results
if not os.path.exists(os.path.join(LOG_DIR, 'test')):
    os.makedirs(os.path.join(LOG_DIR, 'test'))

#create directory for logs 
if not os.path.exists(os.path.join(LOG_DIR, 'logs')):
    os.makedirs(os.path.join(LOG_DIR, 'logs'))
text_file = os.path.join(LOG_DIR, 'log_train.txt')
text_file2 = os.path.join(LOG_DIR, 'log_std_output.txt')


import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(text_file2,"a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  

sys.stdout = Logger()

LOG_FOUT = open(text_file, 'w')

print(args)
LOG_FOUT.write(str(args)+'\n')


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
        print(out_str)


log_string('WANB Project name', wandb_run_name)
log_string('Saving models to ', MODEL_DIR)

log_string('==> Writing text file and stdout pushing file output to ')
log_string(text_file)
log_string(text_file2)

# LOG_DIR = os.path.join('logs', args.log_dir)
# args.log_dir = LOG_DIR

#summary writer
tr_writer = SummaryWriter(LOG_DIR)
val_writer = SummaryWriter(os.path.join(LOG_DIR, 'validation'))
te_writer = SummaryWriter(os.path.join(LOG_DIR, 'test'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.tensorboard.patch(root_logdir=LOG_DIR)

cpu_workers = 4

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
#init parameters 

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         #dont conv, just linear
#         if isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-2)
#             if m.bias is not None:
#                 init.constant(m.bias, 0)

# def transformer_init(transformer):
#     #initlize the transformer
#     for m in transformer.modules():
#         if isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-2)
#             if m.bias is not None:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             init.constant(m.bias, 0)
#             init.constant(m.weight, 1.0)


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

best_acc = 0  # best test accuracy
best_valid_acc = 0 # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


decay_inter = [250, 450]

# Data
print('==> Preparing data..')

if args.dataset == 'PACS':
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    # ctx_test = 2 * ctx_num
    ctx_test = ctx_num
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = PACS(test_domain)
    rt_context = rtPACS(test_domain, ctx_num)
else:
    raise NotImplementedError

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird



print('--> --> LOG_DIR <-- <--', LOG_DIR)



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# #load resnet18 from dg_model    
# if args.model == 'resnet18':
#     net = dg_model.resnet18(pretrained=True, num_classes=NUM_CLASS)
# elif args.model == 'resnet50':
#     net = dg_model.resnet34(pretrained=True, num_classes=NUM_CLASS)
#obtain resenet18 from dg_model 
# net = dg_model.ResNet18() 
# net = dg_meta_model.ResNet18()
# net = dg_model_tr_net.ResNet18_tr_net()
# net = dg_model_layer67.ResNet18()
import dg_model_layer_67_fc
net = dg_model_layer_67_fc.ResNet18()

# tr_net = transformer_model_confg_67.TransformerModel_67(feature_dim=512, num_class= 8)
#give transofmer only the feautres 
# import transofmer_model_feautres
# tr_net = transofmer_model_feautres.TransformerModel_67(feature_dim=512, num_class= 10,  nlayers=tr_net_layers)
# tr_net_gen2 = transofmer_model_feautres.TransformerModel_67(feature_dim=512, num_class= 10,  nlayers=tr_net_layers)
import transofmer_model_feautres
tr_net_fc = transofmer_model_feautres.TransformerModel_67(feature_dim=512, num_class= 10,  nlayers=tr_net_layers)
# tr_net_gen2 = transofmer_model_feautres.TransformerModel_67(feature_dim=512, num_class= 10,  nlayers=tr_net_layers)
#define two optimizers 
# Model
print('==> Building model..')
# print(net)
#net inits 
net.apply(inplace_relu)



net = net.to(device)
tr_net_fc = tr_net_fc.to(device)
# tr_net = tr_net.to(device)
# tr_net_gen2 = tr_net_gen2.to(device)

# print('The number of gradient layers are ')
# for name, param in net.named_parameters():
#   #print layer and if it requires grad
#     print(name, param.requires_grad)

# #print all  the layer names and shape 
# for name, param in net.named_parameters():
#     print(name, param.shape)

# #print all Batch norm layers
# for name, param in net.named_parameters():
#     if 'bn' in name:
#         print(name, param.shape)

#list of all BNs 
# 4th layer, 4.0 features.4.0.bn1.weight torch.Size([64]) features.4.0.bn1.bias torch.Size([64]) features.4.0.bn2.weight torch.Size([64]) features.4.0.bn2.bias torch.Size([64]) 
# 4th layer, 4.1 features.4.1.bn1.weight torch.Size([64]) features.4.1.bn1.bias torch.Size([64]) features.4.1.bn2.weight torch.Size([64]) features.4.1.bn2.bias torch.Size([64]) 
# 5th layer, 5.0 features.5.0.bn1.weight torch.Size([128]) features.5.0.bn1.bias torch.Size([128]) features.5.0.bn2.weight torch.Size([128]) features.5.0.bn2.bias torch.Size([128]) 
# 5th layer,  5.1 features.5.1.bn1.weight torch.Size([128]) features.5.1.bn1.bias torch.Size([128]) features.5.1.bn2.weight torch.Size([128]) features.5.1.bn2.bias torch.Size([128]) 
# 6th layer, 6.0 features.6.0.bn1.weight torch.Size([256]) features.6.0.bn1.bias torch.Size([256]) features.6.0.bn2.weight torch.Size([256]) features.6.0.bn2.bias torch.Size([256]) 
#6th layer, 6.1 features.6.1.bn1.weight torch.Size([256]) features.6.1.bn1.bias torch.Size([256]) features.6.1.bn2.weight torch.Size([256]) features.6.1.bn2.bias torch.Size([256]) 
#7th layer 7.0 features.7.0.bn1.weight torch.Size([512]) features.7.0.bn1.bias torch.Size([512]) features.7.0.bn2.weight torch.Size([512]) features.7.0.bn2.bias torch.Size([512])
#7th layer 7.1 features.7.1.bn1.weight torch.Size([512]) features.7.1.bn1.bias torch.Size([512]) features.7.1.bn2.weight torch.Size([512]) features.7.1.bn2.bias torch.Size([512])



# exit()


# exit()

# # #multi gpu not needed for now 
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # net = nn.DataParallel(net)
#     tr_net = nn.DataParallel(tr_net)


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
# net = nn.DataParallel(net)
# tr_net = nn.DataParallel(tr_net)
# import pdb; pdb.set_trace()





pc = get_parameter_number(net)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

'''old 
#count number of batch norm layers
bn_count = 0
net.eval()
with torch.no_grad():
        
    for class1,m in net.features.named_modules():
        # if isinstance(m, nn.BatchNorm2d):
        #check if its a batch norm layer
        #check if its a batch norm layer
        if isinstance(m, nn.BatchNorm2d):

        
        # if 1==1:
            print(m)
            bn_count += 1
            # print(m)
            # print(m.state_dict().keys())
            #get the batch norm name 
            import pdb
            pdb.set_trace()
            bn_name = m[0]

            
            layer_values = m.state_dict()
            print('Before weights are: ', m.state_dict()['weight'])
            new_wt = torch.zeros_like(layer_values['weight'])
            #layer_values['weight'] 
            #asssign it as new weight of the layer
            print('created weight: ', new_wt)
            m.state_dict()['weight'] =  torch.nn.parameter.Parameter(new_wt)
            print('After weights are: ', m.state_dict()['weight'])
            exit()'''


# for (name, layer) in net.features._modules.items():
#     #iteration over outer layers
#     print((name, layer))
    
#ptint model layer weights shape and layer name 
# for (name, layer) in net.features._modules.items():
#named modules 
# for (name, layer) in net.named_modules():
#     if isinstance(layer, nn.Conv2d):
#         # print((name, layer))
#         # print(layer.weight.shape)
#         # print(layer.bias.shape)
#         print('Name of the layer: ', name, 'Shape of the layer: ', layer.weight.shape)
#     # print('')
#     # print('Name of the layer: ', name, 'Shape of the layer: ', layer.weight.shape)

# print('------------------------')

# for layer in net.modules():
#    if isinstance(layer, nn.Linear):
#         print(layer.weight.shape)    
# #access every layer of resnet and print their name and current weights
# for layer in net.modules():
#     if isinstance(layer, nn.Conv2d):
#         print(layer.weight.shape)
#         # import pdb; pdb.set_trace()
#         # print(layer.bias.shape)
#         print('------------------------')

# #torch summary
# from torchinfo import summary
# b_size_summary = 32

# summary(net, input_size = (b_size_summary, 3, 224, 224)) 

# import pdb
# pdb.set_trace()

# resnet._modules['layer1'][0]._modules['bn1']
#eandb define metric

python_file_name = os.path.basename(__file__)
#get current working directory
current_directory = os.getcwd()
python_file_name = os.path.join(current_directory, python_file_name)
#upload file to wandb 
wandb.save(python_file_name)
log_string('Uploaded file: %s' % python_file_name)


wandb.define_metric("epoch")
wandb.define_metric("meta_target/loss", step_metric="epoch")
wandb.define_metric("meta_target/acc", step_metric="epoch")
wandb.define_metric("meta_target/lr", step_metric="epoch")
wandb.define_metric("meta_target/tr_net_loss", step_metric="epoch")

# wandb.define_metric("epoch")
wandb.define_metric("train/loss", step_metric="epoch")
wandb.define_metric("train/acc", step_metric="epoch")
wandb.define_metric("train/lr", step_metric="epoch")

#meta_target 
wandb.define_metric("meta_target_acc/source_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/source_loss", step_metric="epoch")

wandb.define_metric("meta_target_acc/entropy_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/entropy_loss", step_metric="epoch")


wandb.define_metric("meta_target_acc/model_gen_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/model_gen_loss", step_metric="epoch")

#for transformer gen 2 
wandb.define_metric("meta_target_acc/model_gen2_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/model_gen2_loss", step_metric="epoch")






# wandb.define_metric("epoch")
wandb.define_metric("test/source_acc", step_metric="epoch")
wandb.define_metric("test/source_loss", step_metric="epoch")

wandb.define_metric("test/entropy_acc", step_metric="epoch")
wandb.define_metric("test/entropy_loss", step_metric="epoch")

wandb.define_metric("test/model_gen_acc", step_metric="epoch")
wandb.define_metric("test/model_gen_loss", step_metric="epoch")

#for transformer gen 2
wandb.define_metric("test/model_gen2_acc", step_metric="epoch")
wandb.define_metric("test/model_gen2_loss", step_metric="epoch")



wandb.define_metric("normal_test/acc", step_metric="epoch")
wandb.define_metric("normal_test/loss", step_metric="epoch")

#conv layer trakcer 
conv_layer_list = ['features.0', 'features.4.0.conv1', 'features.4.0.conv2', 'features.4.1.conv1', 'features.4.1.conv2', 'features.5.0.conv1', 'features.5.0.conv2', 'features.5.0.downsample.0', 'features.5.1.conv1', 'features.5.1.conv2', 'features.6.0.conv1', 'features.6.0.conv2', 'features.6.0.downsample.0', 'features.6.1.conv1', 'features.6.1.conv2', 'features.7.0.conv1', 'features.7.0.conv2', 'features.7.0.downsample.0', 'features.7.1.conv1', 'features.7.1.conv2']
#define a wandb metric 
for layer in conv_layer_list:
    wandb.define_metric('layer_wise/' + layer, step_metric="epoch")
    # step_metric="epoch")



# # print('number of batch norm layers: ', bn_count)
# import pdb 
# pdb.set_trace()
# exit()

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)


net.train()
tr_net_fc.train()
wandb.watch(net, log="all", log_freq=10)
wandb.watch(tr_net_fc, log="all", log_freq=10)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#define loss and weight
criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay
decay_inter_optim = [500, 20000, 30000, 40000]


optimizer = torch.optim.AdamW([{'params': net.features.parameters(), 'lr':args.lr * res_lr},   # different lr)
                              {'params': net.fc.parameters(), 'lr':args.lr}], weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.AdamW([{'params': net.parameters(), 'lr':args.lr}], weight_decay=WEIGHT_DECAY)

optimizer_tr_net_fc = torch.optim.AdamW([{'params':tr_net_fc.parameters(), 'lr':0.01}], weight_decay=WEIGHT_DECAY)
# optimizer_tr_net_gen2 = torch.optim.AdamW([{'params':tr_net_gen2.parameters(), 'lr':0.01}], weight_decay=WEIGHT_DECAY)
scheduler_tr_net = torch.optim.lr_scheduler.MultiStepLR(optimizer_tr_net_fc, milestones=decay_inter_optim, gamma=0.1)
# scheduler_tr_net_gen2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_tr_net_gen2, milestones=decay_inter_optim, gamma=0.1)


print('Using optimizer: %s' % optimizer)
# #lr print 
# for param_group in optimizer.param_groups:
#     print(param_group['lr'])
# for param_group in optimizer_tr_net.param_groups:
#     print(param_group['lr'])
# for param_group in optimizer_tr_net_gen2.param_groups:
#     print(param_group['lr'])


if args.resume_from_checkpoint:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.t7')
    # net.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

    checkpoint = torch.load(resume_from_checkpoint)
    net.load_state_dict(checkpoint['net'])
    tr_net.load_state_dict(checkpoint['tr_net'])
    tr_net_gen2.load_state_dict(checkpoint['tr_net_gen2'])
    best_acc = checkpoint['model_gen_acc']
    start_epoch = checkpoint['epoch']
    print('Epoch %d, best_acc %f' % (start_epoch, best_acc))
    
    print('==> Resuming from checkpoint.. done')
    #add sleep for 10 seconds 
    # import time 
    # time.sleep(10)



def compute_layerwise_metrics(net, loader):
    # Only considering "weight" gradients for simplicity
    net.train()
    layer_names = [
        n for n, _ in net.named_parameters() if "bn" not in n
    ]  # and "bias" not in n]

    metrics = defaultdict(list)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y, _ in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits, _ = net(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=net.parameters(), retain_graph=True, allow_unused=True)
        # import pdb; pdb.set_trace()
        # xent_grads.append([g.detach() for g in grad_xent])
        xent_grads.append([g for g in grad_xent])

        # Entropy of predictions. Can calculate without labels.model
        # import Categorical 
        
        loss_entropy = torch.distributions.categorical.Categorical(logits=logits).entropy().mean()
        grad_entropy = torch.autograd.grad(
            outputs=loss_entropy, inputs=net.parameters(), retain_graph=True, allow_unused=True
        )
        entropy_grads.append([g for g in grad_entropy])

    def get_grad_norms(net, grads):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(net.named_parameters(), grads):
            if name not in layer_names:
                continue
            _metrics["grad_norm"].append(torch.norm(grad).item())
            _metrics["rel_grad_norm"].append(
                torch.norm(grad).item() / torch.norm(param).item()
            )
            _metrics["grad_abs"].append(grad.abs().mean().item())
            _metrics["rel_grad_abs"].append((grad.abs() / (param.abs() + 1e-6)).mean().item())
        # import pdb; pdb.set_trace()
        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(net, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[f"xent_{k}"].append(v)
    for entropy_grad in entropy_grads:
        ent_grad_metrics = get_grad_norms(net, entropy_grad)
        for k, v in ent_grad_metrics.items():
            metrics[f"ent_{k}"].append(v)

    # import pdb; pdb.set_trace()

    num_pointwise = min(10, loader.batch_size)
    pt_xent_grads, pt_ent_grads = [], []
    x, y, _ = next(iter(loader))
    x, y = x.cuda(), y.cuda()
    # y = 9-y
    logits, _ = net(x) #model 
    loss_xent_pointwise = F.cross_entropy(logits, y, reduction="none")[:num_pointwise]
    for _loss in loss_xent_pointwise:
        grad_xent_pt = torch.autograd.grad(
            outputs=_loss, inputs=net.parameters(), retain_graph=True
        )
        pt_xent_grads.append([g.detach() for g in grad_xent_pt])
    #import Catagorical
    from torch.distributions import Categorical
    loss_ent_pointwise = Categorical(logits=logits).entropy()[:num_pointwise]
    for _loss in loss_ent_pointwise:
        grad_ent_pt = torch.autograd.grad(
            outputs=_loss, inputs=net.parameters(), retain_graph=True
        )
        pt_ent_grads.append([g.detach() for g in grad_ent_pt])

    def get_pointwise_grad_norms(net, grads):
        all_cosine_sims = []
        for grads1, grads2 in itertools.combinations(grads, 2):
            cosine_sims = []
            for (name, _), g1, g2 in zip(net.named_parameters(), grads1, grads2):
                if name not in layer_names:
                    continue
                cosine_sims.append(
                    F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
                )
            all_cosine_sims.append(cosine_sims)
        return all_cosine_sims

    metrics["xent_pairwise_cosine_sim"] = get_pointwise_grad_norms(net, pt_xent_grads)
    metrics["ent_pairwise_cosine_sim"] = get_pointwise_grad_norms(net, pt_ent_grads)
    from matplotlib import pyplot as plt

    # for k, v in metrics.items():
    #     average_layerwise_metric = np.array(v).mean(0)
    #     plt.plot(range(len(average_layerwise_metric)), average_layerwise_metric, label=k)
    #     plt.xlabel("Layer")
    #     plt.title(f"{k}")
    #     wandb.log({f"plots/{k}": wandb.Image(plt)}, commit=False)
    #     plt.cla()

from collections import defaultdict
import itertools



def get_lr_weights(net, loader):
    # Only considering "weight" gradients for simplicity
    layer_names = [
        n for n, _ in net.named_parameters() if "bn" not in n
    ]  # and "bias" not in n]

    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y, _ in partial_loader:
        x, y = x.cuda(), y.cuda()
        # if cfg.args.flip_labels:
            # y = 9-y  # Reverse labels; quick way to simulate last-layer setting
        logits, _ = net(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=net.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def get_grad_norms(net, grads):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(net.named_parameters(), grads):
            if name not in layer_names:
                continue
            if 1==2:
                tmp = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
                _metrics[name] = tmp.mean().item()
            else:
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(net, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics


def train(epoch):
    
    

    # wandb.define_metric("train/kl_loss", step_metric="epoch")
    # wandb.define_metric("train/w_loss", step_metric="epoch")
    # wandb.define_metric("train/js_div_loss", step_metric="epoch")
    # wandb.define_metric("train/adapt_loss", step_metric = "epoch")
    # all_dataset.reset('train', domain_id, transform=transform_train)

    # kl_loss_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    # rt_context.reset('train', domain_id, transform=transform_train)
    
    # log_string('\nEpoch: %d' % epoch)
    net.train()
    #2 transformers set to eval mode since they are not being used 
    tr_net_fc.eval()
    # tr_net_gen2.eval()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    kl_loss_tot = 0
    w_loss_tot = 0
    js_div_tot= 0
    correct_source = 0
    total_source = 0
    adapt_loss_tot = 0 
    if epoch<3:
        domain_id = epoch
        loss_rate = 1e-8
    else:
        random.seed(6465)
        domain_id = np.random.randint(len(domains))
        loss_rate = 1
    log_string('\n ')
    log_string('Domain ID %d' % domain_id)
    #instead of all dataset use
    all_dataset.reset('train', domain_id, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_workers, drop_last=False, worker_init_fn= worker_init_fn )
    # kl_loss_criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    # rt_context.reset('train', domain_id, transform=transform_train)
    # context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_num, shuffle=False, num_workers=cpu_workers, drop_last=False, worker_init_fn=worker_init_fn)
    # print(time.time()-t0)
    #open a notepad file for logging
    # f1 = open((os.path.join(LOG_DIR, 'log_labels.txt')), 'a')

    # for batch_idx, (inputs, targets, img_name1 ) in enumerate(context_loader):
    #     context_img, context_label = inputs.to(device), targets.to(device)
    #     #train 
    # compute_layerwise_metrics(net, trainloader)

    ''' Commenting SF now since the blocks are already known
    SF says FC is also being updated, certianly because it is from scratch and resnet is pretrained 
    layer_weights = [0 for layer, _ in net.named_parameters() if 'bn' not in layer]
    layer_names = [layer for layer, _ in net.named_parameters() if 'bn' not in layer]
    compute_layerwise_metrics(net, trainloader)
    weights = get_lr_weights(net, trainloader)
    max_weight = max(weights.values())
    for k, v in weights.items(): 
        weights[k] = v / max_weight
    layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
    #code after it 
    tune_metrics = defaultdict(list)
    lr = 0.001


    tune_metrics['layer_weights'] = layer_weights
    params = defaultdict()
    for n, p in net.named_parameters():
        if "bn" not in n:
            params[n] = p 
    params_weights = []
    for param, weight in weights.items():
        params_weights.append({"params": params[param], "lr": weight*lr})
    # opt = optim.AdamW(params_weights, lr=lr, weight_decay=wd)

    import pdb; pdb.set_trace()'''


    for batch_idx, (inputs, targets, img_name2 ) in enumerate(trainloader): 
        #test 
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        #outout format is ctx, x 
        #do pseudo label for trainloader, targets 
        # outputs_ul, outputs  = net(inputs, context_img )
        outputs, _ = net(inputs)
        #print('outputs and targets shape ', outputs.shape, targets.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        #isert gradient tracking here 
        scores = {}
        for name, m in net.named_modules():
        #     if isinstance(m, GraphConv):
            if isinstance(m, torch.nn.Conv2d):
                scores[name] = torch.clone(m.weight.grad.clone()).detach()
            if isinstance(m, torch.nn.BatchNorm2d):
                #take both gradients with weight and bias as keys 
                # scores[name] = torch.clone(m.weight.grad.clone()).detach()
                scores[name + '_weight'] = torch.clone(m.weight.grad.clone()).detach()
                scores[name + '_bias'] = torch.clone(m.bias.grad.clone()).detach()

                # scores[name] = torch.clone(m.bias.grad.clone()).detach()
                # import  pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        # import pdb; pdb.set_trace()
        grad_flow = torch.norm(all_scores)





        #do optimizer step here after tracking gradients flow 
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

       
        if iteration_training and batch_idx>=batchs_per_epoch:
            break
        
    #log_string
    log_string('Epoch: %d, Loss: %.3f, Acc: %.3f' % (epoch, train_loss/(batch_idx+1), 100.*correct/total))
    # log_string('\t Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
   
    #writer 
    wandb.log({'train/loss': train_loss/(batch_idx+1)})
    wandb.log({'epoch':epoch})
    wandb.log({'train/acc': 100.*correct/total})
    wandb.log({'train/lr': optimizer.param_groups[0]['lr']})


    # grad_update_wandb = {}
    # for key, value in scores.items():
    #     #compute norm of the value and update
    #     new_value = torch.norm(value)
    #     # import pdb; pdb.set_trace()
    #     grad_update_wandb[key] = new_value
    #     key = str(key)
    #     wandb.log({'layer_wise/' + key: new_value, 'epoch': epoch})
    # #from grad_update_wandb dict, print the key with max value
    # max_key = max(grad_update_wandb, key=grad_update_wandb.get)
    # log_string('Max grad flow layer: %s' % max_key)



    print('time elapsed: %f' % (time.time()-t0))


    return domain_id, scores 

# criterion_numerical = torch.nn.MSELoss()
#use huber loss 
criterion_numerical = torch.nn.HuberLoss()

def fc_extractor_all(model):
    d1_weights = []
    d1_bias = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc_layer_list = ['fc.weight']
    size_dict = {}
    for i in fc_layer_list:
        if 'weight' in i:
            d1_weights.append(model.state_dict()[i].detach().clone().to(device))
            size_dict[i] = model.state_dict()[i].shape[-1]
        # elif 'bias' in i:
        #     d1_bias.append(model.state_dict()[i].detach().clone().to(device))
        #     size_dict[i] = model.state_dict()[i].shape[-1]
    # import pdb; pdb.set_trace()
    all_weight_tensors = d1_weights
    # all_bias_tensors = d1_bias
    target_size = 512
    weights = torch.stack(all_weight_tensors)
    # bias = torch.stack(all_bias_tensors)
    #move to device
    # weights = weights[0].to(device)
    weights = weights.squeeze()
    weights = weights.to(device)
    # bias = bias.to(device)
    #reshape weights to 512
    return weights


'''
def bn_extractor(net):
    bn_list = []
    weights = []
    bias = []

    # net.features._modules['6']._modules['0']._modules['bn1'].weight
    # net.features._modules['6']._modules['0']._modules['bn1'].bias
    # net.features._modules['6']._modules['0']._modules['bn2'].weight
    # net.features._modules['6']._modules['0']._modules['bn2'].bias


    # net.features._modules['6']._modules['1']._modules['bn1'].weight
    # net.features._modules['6']._modules['1']._modules['bn1'].bias
    # net.features._modules['6']._modules['1']._modules['bn2'].weight
    # net.features._modules['6']._modules['1']._modules['bn2'].bias



    # net.features._modules['7']._modules['0']._modules['bn1'].weight
    # net.features._modules['7']._modules['0']._modules['bn1'].bias
    # net.features._modules['7']._modules['0']._modules['bn2'].weight
    # net.features._modules['7']._modules['0']._modules['bn2'].bias

    # net.features._modules['7']._modules['1']._modules['bn1'].weight
    # net.features._modules['7']._modules['1']._modules['bn1'].bias
    # net.features._modules['7']._modules['1']._modules['bn2'].weight
    # net.features._modules['7']._modules['1']._modules['bn2'].bias
    #create arrays 


    a1 = net.features._modules['6']._modules['0']._modules['bn1'].weight
    a2 = net.features._modules['6']._modules['0']._modules['bn1'].bias
    a3 = net.features._modules['6']._modules['0']._modules['bn2'].weight
    a4 = net.features._modules['6']._modules['0']._modules['bn2'].bias

    a5 = net.features._modules['6']._modules['1']._modules['bn1'].weight
    a6 = net.features._modules['6']._modules['1']._modules['bn1'].bias
    a7 = net.features._modules['6']._modules['1']._modules['bn2'].weight
    a8 = net.features._modules['6']._modules['1']._modules['bn2'].bias

    a9 = net.features._modules['7']._modules['0']._modules['bn1'].weight
    a10 = net.features._modules['7']._modules['0']._modules['bn1'].bias
    a11 = net.features._modules['7']._modules['0']._modules['bn2'].weight
    a12 = net.features._modules['7']._modules['0']._modules['bn2'].bias

    a13 = net.features._modules['7']._modules['1']._modules['bn1'].weight
    a14 = net.features._modules['7']._modules['1']._modules['bn1'].bias
    a15 = net.features._modules['7']._modules['1']._modules['bn2'].weight
    a16 = net.features._modules['7']._modules['1']._modules['bn2'].bias


    all_tensors = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]
    target_size = 512

    for i, tensor in enumerate(all_tensors):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_tensors[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0)
    
    weights_bias = torch.stack(all_tensors)

    # import pdb; pdb.set_trace()
    return weights_bias

'''

#seperate weights and bias

# def bn_extractor_all(model):
    
#     d1_weights = []
#     d1_bias = []
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     #collect all BNs from 4th layer to 7th layer net
#     #list of all BN layers to be collected 
#     # bn_layer_list = ['features.1.weight', 'features.1.bias', 'features.4.0.bn1.weight', 'features.4.0.bn1.bias', 'features.4.0.bn2.weight', 'features.4.0.bn2.bias', 'features.4.1.bn1.weight', 'features.4.1.bn1.bias', 'features.4.1.bn2.weight', 'features.4.1.bn2.bias', 'features.5.0.bn1.weight', 'features.5.0.bn1.bias', 'features.5.0.bn2.weight', 'features.5.0.bn2.bias', 'features.5.0.downsample.1.weight', 'features.5.0.downsample.1.bias', 'features.5.1.bn1.weight', 'features.5.1.bn1.bias', 'features.5.1.bn2.weight', 'features.5.1.bn2.bias', 'features.6.0.bn1.weight', 'features.6.0.bn1.bias', 'features.6.0.bn2.weight', 'features.6.0.bn2.bias', 'features.6.0.downsample.1.weight', 'features.6.0.downsample.1.bias', 'features.6.1.bn1.weight', 'features.6.1.bn1.bias', 'features.6.1.bn2.weight', 'features.6.1.bn2.bias', 'features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias', 'features.7.0.downsample.1.weight', 'features.7.0.downsample.1.bias', 'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']

#     bn_layer_list = ['features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias',  'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']

#     #collect all Bns from bn_layer_list
#     size_dict = {} 
#     for i in bn_layer_list:
#         #collect weight and bias seperately
#         if 'weight' in i:
#             #detach and then copy 
#             d1_weights.append(model.state_dict()[i].detach().clone().to(device))
#             size_dict[i] = model.state_dict()[i].shape[-1]
#         elif 'bias' in i:
#             d1_bias.append(model.state_dict()[i].detach().clone().to(device))
#             size_dict[i] = model.state_dict()[i].shape[-1]
    
        


#     all_weight_tensors = d1_weights
    
#     all_bias_tensors = d1_bias
#     target_size = 512
#     # import pdb; pdb.set_trace()

    

#     weights = torch.stack(all_weight_tensors)
#     bias = torch.stack(all_bias_tensors)
#     #move to device 
#     weights = weights.to(device)
#     bias = bias.to(device)
#     # import pdb; pdb.set_trace()
#     # import pdb ; pdb.set_trace()


#     return weights, bias

def fc_extractor_all(model):
    d1_weights = []
    d1_bias = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc_layer_list = ['fc.weight']
    size_dict = {}
    for i in fc_layer_list:
        if 'weight' in i:
            d1_weights.append(model.state_dict()[i].detach().clone().to(device))
            size_dict[i] = model.state_dict()[i].shape[-1]
        # elif 'bias' in i:
        #     d1_bias.append(model.state_dict()[i].detach().clone().to(device))
        #     size_dict[i] = model.state_dict()[i].shape[-1]
    # import pdb; pdb.set_trace()
    all_weight_tensors = d1_weights
    # all_bias_tensors = d1_bias
    target_size = 512
    weights = torch.stack(all_weight_tensors)
    # bias = torch.stack(all_bias_tensors)
    #move to device
    # weights = weights[0].to(device)
    weights = weights.squeeze()
    weights = weights.to(device)
    # bias = bias.to(device)
    #reshape weights to 512
    return weights


def weights_generator_fc(model, outputs_tr_net):
    #size of gradients {'features.1.weight': 64, 'features.1.bias': 64, 'features.4.0.bn1.weight': 64, 'features.4.0.bn1.bias': 64, 'features.4.0.bn2.weight': 64, 'features.4.0.bn2.bias': 64, 'features.4.1.bn1.weight': 64, 'features.4.1.bn1.bias': 64, 'features.4.1.bn2.weight': 64, 'features.4.1.bn2.bias': 64, 'features.5.0.bn1.weight': 128, 'features.5.0.bn1.bias': 128, 'features.5.0.bn2.weight': 128, 'features.5.0.bn2.bias': 128, 'features.5.0.downsample.1.weight': 128, 'features.5.0.downsample.1.bias': 128, 'features.5.1.bn1.weight': 128, 'features.5.1.bn1.bias': 128, 'features.5.1.bn2.weight': 128, 'features.5.1.bn2.bias': 128, 'features.6.0.bn1.weight': 256, 'features.6.0.bn1.bias': 256, 'features.6.0.bn2.weight': 256, 'features.6.0.bn2.bias': 256, 'features.6.0.downsample.1.weight': 256, 'features.6.0.downsample.1.bias': 256, 'features.6.1.bn1.weight': 256, 'features.6.1.bn1.bias': 256, 'features.6.1.bn2.weight': 256, 'features.6.1.bn2.bias': 256, 'features.7.0.bn1.weight': 512, 'features.7.0.bn1.bias': 512, 'features.7.0.bn2.weight': 512, 'features.7.0.bn2.bias': 512, 'features.7.0.downsample.1.weight': 512, 'features.7.0.downsample.1.bias': 512, 'features.7.1.bn1.weight': 512, 'features.7.1.bn1.bias': 512, 'features.7.1.bn2.weight': 512, 'features.7.1.bn2.bias': 512}
    added_weights = []
    added_bias = []
    #layer 7
    # added_weights.append(new_gamma[0][:512] + model.features._modules['7']._modules['0']._modules['bn1'].weight.detach())
    #for FC layer in net 
    #reshape needed add extra dimenion if needed. 
    # added_weights.append(outputs_tr_net)
    added_weights.append(model.fc.weight + outputs_tr_net)
    # added_bias = added_bias.to(device)
    added_weights = torch.stack(added_weights)
    added_weights = added_weights.to(device)
    added_weights = added_weights.squeeze(0)
    with torch.no_grad():
    # import pdb; pdb.set_trace()
    # if 1==1:
    
        
        #fc layer 
        model.fc.weight += nn.Parameter(outputs_tr_net)
        
    return model, added_weights


def gradients_extract_fc(d1):
    weights = []
    # detatch first
    d1 = {k: v.detach() for k, v in d1.items()}
    all_weights = []
    all_bias = []
    for k, v in d1.items():
        # check if bn and weight is present
        #check if downsample is not present
        if 'fc' in k:
            if 'weight' in k:
                all_weights.append(v)
                # print('Key and value are: ', k, v)
                # print('Key is : ', k)
    
    # import pdb; pdb.set_trace()
            
    # all_weight_gradients = all_weights
    # target_size = 512
    # for i, tensor in enumerate(all_weight_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size
    #     if pad_size > 0:
    #         all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)
    # all_bias_gradients = all_bias
    # for i, tensor in enumerate(all_bias_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size
    #     if pad_size > 0:
    #         all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)
    # weights = torch.stack(all_weight_gradients, 0).to(device)
    # bias = torch.stack(all_bias_gradients, 0).to(device)
    all_weight_gradients = torch.stack(all_weights).to(device)
    all_weight_gradients = all_weight_gradients.squeeze()
    # import pdb; pdb.set_trace()
    
    # all_bias_gradients = torch.stack(all_bias).to(device)
    
    # import pdb; pdb.set_trace()
    # return weights, bias
    return all_weight_gradients


def bn_extractor_gen2(model):
    
    d1_weights = []
    d1_bias = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #collect all BNs from 4th layer to 7th layer net
    #list of all BN layers to be collected 
    # bn_layer_list = ['features.1.weight', 'features.1.bias', 'features.4.0.bn1.weight', 'features.4.0.bn1.bias', 'features.4.0.bn2.weight', 'features.4.0.bn2.bias', 'features.4.1.bn1.weight', 'features.4.1.bn1.bias', 'features.4.1.bn2.weight', 'features.4.1.bn2.bias', 'features.5.0.bn1.weight', 'features.5.0.bn1.bias', 'features.5.0.bn2.weight', 'features.5.0.bn2.bias', 'features.5.0.downsample.1.weight', 'features.5.0.downsample.1.bias', 'features.5.1.bn1.weight', 'features.5.1.bn1.bias', 'features.5.1.bn2.weight', 'features.5.1.bn2.bias', 'features.6.0.bn1.weight', 'features.6.0.bn1.bias', 'features.6.0.bn2.weight', 'features.6.0.bn2.bias', 'features.6.0.downsample.1.weight', 'features.6.0.downsample.1.bias', 'features.6.1.bn1.weight', 'features.6.1.bn1.bias', 'features.6.1.bn2.weight', 'features.6.1.bn2.bias', 'features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias', 'features.7.0.downsample.1.weight', 'features.7.0.downsample.1.bias', 'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']

    bn_layer_list = ['features.1.weight', 'features.1.bias', 'features.4.0.bn1.weight', 'features.4.0.bn1.bias', 'features.4.0.bn2.weight', 'features.4.0.bn2.bias', 'features.4.1.bn1.weight', 'features.4.1.bn1.bias', 'features.4.1.bn2.weight', 'features.4.1.bn2.bias', 'features.5.0.bn1.weight', 'features.5.0.bn1.bias', 'features.5.0.bn2.weight', 'features.5.0.bn2.bias', 'features.5.0.downsample.1.weight', 'features.5.0.downsample.1.bias', 'features.5.1.bn1.weight', 'features.5.1.bn1.bias', 'features.5.1.bn2.weight', 'features.5.1.bn2.bias', 'features.6.0.bn1.weight', 'features.6.0.bn1.bias', 'features.6.0.bn2.weight', 'features.6.0.bn2.bias', 'features.6.0.downsample.1.weight', 'features.6.0.downsample.1.bias', 'features.6.1.bn1.weight', 'features.6.1.bn1.bias', 'features.6.1.bn2.weight', 'features.6.1.bn2.bias']
    #remove downsample layer from bn_layer_list
    # for i in bn_layer_list:
    #     if 'downsample' in i:
    #         bn_layer_list.remove(i)


    # bn_layer_list = ['features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias',  'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']

    #collect all Bns from bn_layer_list
    size_dict = {} 
    for i in bn_layer_list:
        #collect weight and bias seperately
        if not 'downsample' in i:
            if 'weight' in i:
                #detach and then copy 
                d1_weights.append(model.state_dict()[i].detach().clone().to(device))
                size_dict[i] = model.state_dict()[i].shape[-1]
            elif 'bias' in i:
                d1_bias.append(model.state_dict()[i].detach().clone().to(device))
                size_dict[i] = model.state_dict()[i].shape[-1]
    
        


    all_weight_tensors = d1_weights
    
    all_bias_tensors = d1_bias
    target_size = 512
    # import pdb; pdb.set_trace()

    weights = padding_adder(all_weight_tensors)
    bias = padding_adder(all_bias_tensors)
    # print('weights shape', weights.shape)
    # print('bias shape', bias.shape)


    

    # weights = torch.stack(all_weight_tensors)
    # bias = torch.stack(all_bias_tensors)
    # #move to device 
    # weights = weights.to(device)
    # bias = bias.to(device)
    # import pdb; pdb.set_trace()
    # import pdb ; pdb.set_trace()


    return weights, bias


# #loop for transformer 2 

# def bn_extractor_gen2(model):
    
#     d1_weights = []
#     d1_bias = []
    
    

#     #collect all BNs from 4th layer to 7th layer net
#     #list of all BN layers to be collected 
#     # bn_layer_list = ['features.1.weight', 'features.1.bias', 'features.4.0.bn1.weight', 'features.4.0.bn1.bias', 'features.4.0.bn2.weight', 'features.4.0.bn2.bias', 'features.4.1.bn1.weight', 'features.4.1.bn1.bias', 'features.4.1.bn2.weight', 'features.4.1.bn2.bias', 'features.5.0.bn1.weight', 'features.5.0.bn1.bias', 'features.5.0.bn2.weight', 'features.5.0.bn2.bias', 'features.5.0.downsample.1.weight', 'features.5.0.downsample.1.bias', 'features.5.1.bn1.weight', 'features.5.1.bn1.bias', 'features.5.1.bn2.weight', 'features.5.1.bn2.bias', 'features.6.0.bn1.weight', 'features.6.0.bn1.bias', 'features.6.0.bn2.weight', 'features.6.0.bn2.bias', 'features.6.0.downsample.1.weight', 'features.6.0.downsample.1.bias', 'features.6.1.bn1.weight', 'features.6.1.bn1.bias', 'features.6.1.bn2.weight', 'features.6.1.bn2.bias', 'features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias', 'features.7.0.downsample.1.weight', 'features.7.0.downsample.1.bias', 'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']
#     #remove 7th layer keep others 
#     bn_layer_list = ['features.1.weight', 'features.1.bias', 'features.4.0.bn1.weight', 'features.4.0.bn1.bias', 'features.4.0.bn2.weight', 'features.4.0.bn2.bias', 'features.4.1.bn1.weight', 'features.4.1.bn1.bias', 'features.4.1.bn2.weight', 'features.4.1.bn2.bias', 'features.5.0.bn1.weight', 'features.5.0.bn1.bias', 'features.5.0.bn2.weight', 'features.5.0.bn2.bias', 'features.5.0.downsample.1.weight', 'features.5.0.downsample.1.bias', 'features.5.1.bn1.weight', 'features.5.1.bn1.bias', 'features.5.1.bn2.weight', 'features.5.1.bn2.bias', 'features.6.0.bn1.weight', 'features.6.0.bn1.bias', 'features.6.0.bn2.weight', 'features.6.0.bn2.bias', 'features.6.0.downsample.1.weight', 'features.6.0.downsample.1.bias', 'features.6.1.bn1.weight', 'features.6.1.bn1.bias', 'features.6.1.bn2.weight', 'features.6.1.bn2.bias']  


#     # bn_layer_list = ['features.7.0.bn1.weight', 'features.7.0.bn1.bias', 'features.7.0.bn2.weight', 'features.7.0.bn2.bias',  'features.7.1.bn1.weight', 'features.7.1.bn1.bias', 'features.7.1.bn2.weight', 'features.7.1.bn2.bias']

#     #collect all Bns from bn_layer_list
#     size_dict = {} 
#     for i in bn_layer_list:
#         #collect weight and bias seperately
#         if not 'downsample' in i:
#             if 'weight' in i:
#                 #detach and then copy 
#                 d1_weights.append(model.state_dict()[i].detach().clone().to(device))
#                 size_dict[i] = model.state_dict()[i].shape[-1]
#             elif 'bias' in i:
#                 d1_bias.append(model.state_dict()[i].detach().clone().to(device))
#                 size_dict[i] = model.state_dict()[i].shape[-1]
        
#     print('Number of BNs collected: ', len(d1_weights), len(d1_bias))


#     all_weight_gradients = d1_weights
    
#     all_bias_gradients = d1_bias
#     target_size = 512
#     # import pdb; pdb.set_trace()

#     # all_weight_gradients = all_weights
#     target_size = 512

#     for i, tensor in enumerate(all_weight_gradients):
#         current_size = tensor.shape[-1]
#         pad_size = target_size - current_size

#         if pad_size > 0:
#             all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

#     # all_bias_gradients = all_bias

#     for i, tensor in enumerate(all_bias_gradients):
#         current_size = tensor.shape[-1]
#         pad_size = target_size - current_size

#         if pad_size > 0:
#             all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

#     weights = torch.stack(all_weight_gradients, 0).to(device)
#     bias = torch.stack(all_bias_gradients, 0).to(device)

    

#     # weights = torch.stack(all_weight_tensors)
#     # bias = torch.stack(all_bias_tensors)
#     #move to device 
#     weights = weights.to(device)
#     bias = bias.to(device)
#     # import pdb; pdb.set_trace()
#     # import pdb ; pdb.set_trace()


#     return weights, bias




# '''old gamma and beta 

# '''old gamma and beta 
# def weights_generator(net_copy, new_gamma, new_beta):
#     #size of gradients {'features.1.weight': 64, 'features.1.bias': 64, 'features.4.0.bn1.weight': 64, 'features.4.0.bn1.bias': 64, 'features.4.0.bn2.weight': 64, 'features.4.0.bn2.bias': 64, 'features.4.1.bn1.weight': 64, 'features.4.1.bn1.bias': 64, 'features.4.1.bn2.weight': 64, 'features.4.1.bn2.bias': 64, 'features.5.0.bn1.weight': 128, 'features.5.0.bn1.bias': 128, 'features.5.0.bn2.weight': 128, 'features.5.0.bn2.bias': 128, 'features.5.0.downsample.1.weight': 128, 'features.5.0.downsample.1.bias': 128, 'features.5.1.bn1.weight': 128, 'features.5.1.bn1.bias': 128, 'features.5.1.bn2.weight': 128, 'features.5.1.bn2.bias': 128, 'features.6.0.bn1.weight': 256, 'features.6.0.bn1.bias': 256, 'features.6.0.bn2.weight': 256, 'features.6.0.bn2.bias': 256, 'features.6.0.downsample.1.weight': 256, 'features.6.0.downsample.1.bias': 256, 'features.6.1.bn1.weight': 256, 'features.6.1.bn1.bias': 256, 'features.6.1.bn2.weight': 256, 'features.6.1.bn2.bias': 256, 'features.7.0.bn1.weight': 512, 'features.7.0.bn1.bias': 512, 'features.7.0.bn2.weight': 512, 'features.7.0.bn2.bias': 512, 'features.7.0.downsample.1.weight': 512, 'features.7.0.downsample.1.bias': 512, 'features.7.1.bn1.weight': 512, 'features.7.1.bn1.bias': 512, 'features.7.1.bn2.weight': 512, 'features.7.1.bn2.bias': 512}

#     # with torch.no_grad():
#     # import pdb; pdb.set_trace()
#     if 1==1:
#         #add to layer 1 
#         net_copy.features._modules['1'].weight = nn.Parameter(new_gamma[0][:64])
#         net_copy.features._modules['1'].bias = nn.Parameter(new_beta[0][:64])


#         #layer 4 
#         net_copy.features._modules['4']._modules['0']._modules['bn1'].weight = nn.Parameter(  new_gamma[1][:64])
#         net_copy.features._modules['4']._modules['0']._modules['bn1'].bias = nn.Parameter(new_beta[1][:64])
#         net_copy.features._modules['4']._modules['0']._modules['bn2'].weight = nn.Parameter(new_gamma[2][:64])
#         net_copy.features._modules['4']._modules['0']._modules['bn2'].bias = nn.Parameter(new_beta[2][:64])

#         net_copy.features._modules['4']._modules['1']._modules['bn1'].weight = nn.Parameter(new_gamma[3][:64])
#         net_copy.features._modules['4']._modules['1']._modules['bn1'].bias = nn.Parameter(new_beta[3][:64])
#         net_copy.features._modules['4']._modules['1']._modules['bn2'].weight = nn.Parameter(new_gamma[4][:64])
#         net_copy.features._modules['4']._modules['1']._modules['bn2'].bias = nn.Parameter(new_beta[4][:64])

#         #layer 5
#         net_copy.features._modules['5']._modules['0']._modules['bn1'].weight = nn.Parameter(new_gamma[5][:128])
#         net_copy.features._modules['5']._modules['0']._modules['bn1'].bias = nn.Parameter(new_beta[5][:128])
#         net_copy.features._modules['5']._modules['0']._modules['bn2'].weight = nn.Parameter(new_gamma[6][:128])
#         net_copy.features._modules['5']._modules['0']._modules['bn2'].bias = nn.Parameter(new_beta[6][:128])
#         net_copy.features._modules['5']._modules['0']._modules['downsample']._modules['1'].weight = nn.Parameter(new_gamma[7][:128])
#         net_copy.features._modules['5']._modules['0']._modules['downsample']._modules['1'].bias = nn.Parameter(new_beta[7][:128])

#         net_copy.features._modules['5']._modules['1']._modules['bn1'].weight = nn.Parameter(new_gamma[8][:128])
#         net_copy.features._modules['5']._modules['1']._modules['bn1'].bias = nn.Parameter(new_beta[8][:128])
#         net_copy.features._modules['5']._modules['1']._modules['bn2'].weight = nn.Parameter(new_gamma[9][:128])
#         net_copy.features._modules['5']._modules['1']._modules['bn2'].bias = nn.Parameter(new_beta[9][:128])

#         #layer 6
#         net_copy.features._modules['6']._modules['0']._modules['bn1'].weight = nn.Parameter(new_gamma[10][:256])
#         net_copy.features._modules['6']._modules['0']._modules['bn1'].bias = nn.Parameter(new_beta[10][:256])
#         net_copy.features._modules['6']._modules['0']._modules['bn2'].weight = nn.Parameter(new_gamma[11][:256])
#         net_copy.features._modules['6']._modules['0']._modules['bn2'].bias = nn.Parameter(new_beta[11][:256])
#         net_copy.features._modules['6']._modules['0']._modules['downsample']._modules['1'].weight = nn.Parameter(new_gamma[12][:256])
#         net_copy.features._modules['6']._modules['0']._modules['downsample']._modules['1'].bias = nn.Parameter(new_beta[12][:256])

#         net_copy.features._modules['6']._modules['1']._modules['bn1'].weight = nn.Parameter(new_gamma[13][:256])
#         net_copy.features._modules['6']._modules['1']._modules['bn1'].bias = nn.Parameter(new_beta[13][:256])
#         net_copy.features._modules['6']._modules['1']._modules['bn2'].weight = nn.Parameter(new_gamma[14][:256])
#         net_copy.features._modules['6']._modules['1']._modules['bn2'].bias = nn.Parameter(new_beta[14][:256])

#         #layer 7 
#         net_copy.features._modules['7']._modules['0']._modules['bn1'].weight = nn.Parameter(new_gamma[15][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn1'].bias = nn.Parameter(new_beta[15][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn2'].weight = nn.Parameter(new_gamma[16][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn2'].bias = nn.Parameter(new_beta[16][:512])
#         net_copy.features._modules['7']._modules['0']._modules['downsample']._modules['1'].weight = nn.Parameter(new_gamma[17][:512])
#         net_copy.features._modules['7']._modules['0']._modules['downsample']._modules['1'].bias = nn.Parameter(new_beta[17][:512])

#         net_copy.features._modules['7']._modules['1']._modules['bn1'].weight = nn.Parameter(new_gamma[18][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn1'].bias = nn.Parameter(new_beta[18][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn2'].weight = nn.Parameter(new_gamma[19][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn2'].bias = nn.Parameter(new_beta[19][:512])

#     return net_copy


# def weights_generator(net_copy, new_gamma, new_beta):
#     #size of gradients {'features.1.weight': 64, 'features.1.bias': 64, 'features.4.0.bn1.weight': 64, 'features.4.0.bn1.bias': 64, 'features.4.0.bn2.weight': 64, 'features.4.0.bn2.bias': 64, 'features.4.1.bn1.weight': 64, 'features.4.1.bn1.bias': 64, 'features.4.1.bn2.weight': 64, 'features.4.1.bn2.bias': 64, 'features.5.0.bn1.weight': 128, 'features.5.0.bn1.bias': 128, 'features.5.0.bn2.weight': 128, 'features.5.0.bn2.bias': 128, 'features.5.0.downsample.1.weight': 128, 'features.5.0.downsample.1.bias': 128, 'features.5.1.bn1.weight': 128, 'features.5.1.bn1.bias': 128, 'features.5.1.bn2.weight': 128, 'features.5.1.bn2.bias': 128, 'features.6.0.bn1.weight': 256, 'features.6.0.bn1.bias': 256, 'features.6.0.bn2.weight': 256, 'features.6.0.bn2.bias': 256, 'features.6.0.downsample.1.weight': 256, 'features.6.0.downsample.1.bias': 256, 'features.6.1.bn1.weight': 256, 'features.6.1.bn1.bias': 256, 'features.6.1.bn2.weight': 256, 'features.6.1.bn2.bias': 256, 'features.7.0.bn1.weight': 512, 'features.7.0.bn1.bias': 512, 'features.7.0.bn2.weight': 512, 'features.7.0.bn2.bias': 512, 'features.7.0.downsample.1.weight': 512, 'features.7.0.downsample.1.bias': 512, 'features.7.1.bn1.weight': 512, 'features.7.1.bn1.bias': 512, 'features.7.1.bn2.weight': 512, 'features.7.1.bn2.bias': 512}

#     with torch.no_grad():
#     # import pdb; pdb.set_trace()
#     # if 1==1:
        
#         #layer 7 
#         net_copy.features._modules['7']._modules['0']._modules['bn1'].weight += nn.Parameter(new_gamma[0][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn1'].bias += nn.Parameter(new_beta[1][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn2'].weight += nn.Parameter(new_gamma[2][:512])
#         net_copy.features._modules['7']._modules['0']._modules['bn2'].bias += nn.Parameter(new_beta[3][:512])
        

#         net_copy.features._modules['7']._modules['1']._modules['bn1'].weight += nn.Parameter(new_gamma[0][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn1'].bias += nn.Parameter(new_beta[1][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn2'].weight += nn.Parameter(new_gamma[2][:512])
#         net_copy.features._modules['7']._modules['1']._modules['bn2'].bias += nn.Parameter(new_beta[3][:512])

#     return net_copy


def padding_adder(weights):
    all_weight_gradients = weights
    target_size = 512

    for i, tensor in enumerate(all_weight_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

    weights = torch.stack(all_weight_gradients, 0).to(device)
    

    return weights

def weights_generator(model, new_gamma, new_beta):
    #size of gradients {'features.1.weight': 64, 'features.1.bias': 64, 'features.4.0.bn1.weight': 64, 'features.4.0.bn1.bias': 64, 'features.4.0.bn2.weight': 64, 'features.4.0.bn2.bias': 64, 'features.4.1.bn1.weight': 64, 'features.4.1.bn1.bias': 64, 'features.4.1.bn2.weight': 64, 'features.4.1.bn2.bias': 64, 'features.5.0.bn1.weight': 128, 'features.5.0.bn1.bias': 128, 'features.5.0.bn2.weight': 128, 'features.5.0.bn2.bias': 128, 'features.5.0.downsample.1.weight': 128, 'features.5.0.downsample.1.bias': 128, 'features.5.1.bn1.weight': 128, 'features.5.1.bn1.bias': 128, 'features.5.1.bn2.weight': 128, 'features.5.1.bn2.bias': 128, 'features.6.0.bn1.weight': 256, 'features.6.0.bn1.bias': 256, 'features.6.0.bn2.weight': 256, 'features.6.0.bn2.bias': 256, 'features.6.0.downsample.1.weight': 256, 'features.6.0.downsample.1.bias': 256, 'features.6.1.bn1.weight': 256, 'features.6.1.bn1.bias': 256, 'features.6.1.bn2.weight': 256, 'features.6.1.bn2.bias': 256, 'features.7.0.bn1.weight': 512, 'features.7.0.bn1.bias': 512, 'features.7.0.bn2.weight': 512, 'features.7.0.bn2.bias': 512, 'features.7.0.downsample.1.weight': 512, 'features.7.0.downsample.1.bias': 512, 'features.7.1.bn1.weight': 512, 'features.7.1.bn1.bias': 512, 'features.7.1.bn2.weight': 512, 'features.7.1.bn2.bias': 512}

    added_weights = []
    added_bias = []

    #layer 7
    added_weights.append(new_gamma[0][:512] + model.features._modules['7']._modules['0']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[0][:512] + model.features._modules['7']._modules['0']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[1][:512] + model.features._modules['7']._modules['0']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[1][:512] + model.features._modules['7']._modules['0']._modules['bn2'].bias.detach())

    added_weights.append(new_gamma[2][:512] + model.features._modules['7']._modules['1']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[2][:512] + model.features._modules['7']._modules['1']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[3][:512] + model.features._modules['7']._modules['1']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[3][:512] + model.features._modules['7']._modules['1']._modules['bn2'].bias.detach())


    #convert to tensor 
    added_weights = torch.stack(added_weights)
    added_bias = torch.stack(added_bias)
    added_weights = added_weights.to(device)
    added_bias = added_bias.to(device)





    with torch.no_grad():
    # import pdb; pdb.set_trace()
    # if 1==1:
    
        
        #layer 7 
        model.features._modules['7']._modules['0']._modules['bn1'].weight += nn.Parameter(new_gamma[0][:512])
        model.features._modules['7']._modules['0']._modules['bn1'].bias += nn.Parameter(new_beta[0][:512])
        model.features._modules['7']._modules['0']._modules['bn2'].weight += nn.Parameter(new_gamma[1][:512])
        model.features._modules['7']._modules['0']._modules['bn2'].bias += nn.Parameter(new_beta[1][:512])
        

        model.features._modules['7']._modules['1']._modules['bn1'].weight += nn.Parameter(new_gamma[2][:512])
        model.features._modules['7']._modules['1']._modules['bn1'].bias += nn.Parameter(new_beta[2][:512])
        model.features._modules['7']._modules['1']._modules['bn2'].weight += nn.Parameter(new_gamma[3][:512])
        model.features._modules['7']._modules['1']._modules['bn2'].bias += nn.Parameter(new_beta[3][:512])


        # #extract weights gradients are to be kept, hence first calculate the summation with gradients 
        # weights = []
        # bias = []
        # weights.append(model.features._modules['7']._modules['0']._modules['bn1'].weight)
        # bias.append(model.features._modules['7']._modules['0']._modules['bn1'].bias)
        # weights.append(model.features._modules['7']._modules['0']._modules['bn2'].weight)
        # bias.append(model.features._modules['7']._modules['0']._modules['bn2'].bias)

        # weights.append(model.features._modules['7']._modules['1']._modules['bn1'].weight)
        # bias.append(model.features._modules['7']._modules['1']._modules['bn1'].bias)
        # weights.append(model.features._modules['7']._modules['1']._modules['bn2'].weight)
        # bias.append(model.features._modules['7']._modules['1']._modules['bn2'].bias)

        


    return model, added_weights, added_bias



def weights_generator_gen2(model, new_gamma, new_beta):
    #size of gradients {'features.1.weight': 64, 'features.1.bias': 64, 'features.4.0.bn1.weight': 64, 'features.4.0.bn1.bias': 64, 'features.4.0.bn2.weight': 64, 'features.4.0.bn2.bias': 64, 'features.4.1.bn1.weight': 64, 'features.4.1.bn1.bias': 64, 'features.4.1.bn2.weight': 64, 'features.4.1.bn2.bias': 64, 'features.5.0.bn1.weight': 128, 'features.5.0.bn1.bias': 128, 'features.5.0.bn2.weight': 128, 'features.5.0.bn2.bias': 128, 'features.5.0.downsample.1.weight': 128, 'features.5.0.downsample.1.bias': 128, 'features.5.1.bn1.weight': 128, 'features.5.1.bn1.bias': 128, 'features.5.1.bn2.weight': 128, 'features.5.1.bn2.bias': 128, 'features.6.0.bn1.weight': 256, 'features.6.0.bn1.bias': 256, 'features.6.0.bn2.weight': 256, 'features.6.0.bn2.bias': 256, 'features.6.0.downsample.1.weight': 256, 'features.6.0.downsample.1.bias': 256, 'features.6.1.bn1.weight': 256, 'features.6.1.bn1.bias': 256, 'features.6.1.bn2.weight': 256, 'features.6.1.bn2.bias': 256, 'features.7.0.bn1.weight': 512, 'features.7.0.bn1.bias': 512, 'features.7.0.bn2.weight': 512, 'features.7.0.bn2.bias': 512, 'features.7.0.downsample.1.weight': 512, 'features.7.0.downsample.1.bias': 512, 'features.7.1.bn1.weight': 512, 'features.7.1.bn1.bias': 512, 'features.7.1.bn2.weight': 512, 'features.7.1.bn2.bias': 512}

    added_weights = []
    added_bias = []
    
    #layer 1 
    added_weights.append(new_gamma[0][:64] + model.features._modules['1'].weight.detach())
    added_bias.append(new_beta[0][:64] + model.features._modules['1'].bias.detach())

    #layer2 
    added_weights.append(new_gamma[1][:64] + model.features._modules['4']._modules['0']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[1][:64] + model.features._modules['4']._modules['0']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[2][:64] + model.features._modules['4']._modules['0']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[2][:64] + model.features._modules['4']._modules['0']._modules['bn2'].bias.detach())

    added_weights.append(new_gamma[3][:64] + model.features._modules['4']._modules['1']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[3][:64] + model.features._modules['4']._modules['1']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[4][:64] + model.features._modules['4']._modules['1']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[4][:64] + model.features._modules['4']._modules['1']._modules['bn2'].bias.detach())

    #layer 3
    added_weights.append(new_gamma[5][:128] + model.features._modules['5']._modules['0']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[5][:128] + model.features._modules['5']._modules['0']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[6][:128] + model.features._modules['5']._modules['0']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[6][:128] + model.features._modules['5']._modules['0']._modules['bn2'].bias.detach())

    added_weights.append(new_gamma[7][:128] + model.features._modules['5']._modules['1']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[7][:128] + model.features._modules['5']._modules['1']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[8][:128] + model.features._modules['5']._modules['1']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[8][:128] + model.features._modules['5']._modules['1']._modules['bn2'].bias.detach())

    #layer 4
    added_weights.append(new_gamma[9][:256] + model.features._modules['6']._modules['0']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[9][:256] + model.features._modules['6']._modules['0']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[10][:256] + model.features._modules['6']._modules['0']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[10][:256] + model.features._modules['6']._modules['0']._modules['bn2'].bias.detach())

    added_weights.append(new_gamma[11][:256] + model.features._modules['6']._modules['1']._modules['bn1'].weight.detach())
    added_bias.append(new_beta[11][:256] + model.features._modules['6']._modules['1']._modules['bn1'].bias.detach())
    added_weights.append(new_gamma[12][:256] + model.features._modules['6']._modules['1']._modules['bn2'].weight.detach())
    added_bias.append(new_beta[12][:256] + model.features._modules['6']._modules['1']._modules['bn2'].bias.detach())


    #do padding 
    added_weights = padding_adder(added_weights)
    added_bias = padding_adder(added_bias)


    with torch.no_grad():
    # import pdb; pdb.set_trace()
    # if 1==1:
        #layer 1
        model.features._modules['1'].weight += nn.Parameter(new_gamma[0][:64])
        model.features._modules['1'].bias += nn.Parameter(new_beta[0][:64])

        #layer 2
        model.features._modules['4']._modules['0']._modules['bn1'].weight += nn.Parameter(new_gamma[1][:64])
        model.features._modules['4']._modules['0']._modules['bn1'].bias += nn.Parameter(new_beta[1][:64])
        model.features._modules['4']._modules['0']._modules['bn2'].weight += nn.Parameter(new_gamma[2][:64])
        model.features._modules['4']._modules['0']._modules['bn2'].bias += nn.Parameter(new_beta[2][:64])

        model.features._modules['4']._modules['1']._modules['bn1'].weight += nn.Parameter(new_gamma[3][:64])
        model.features._modules['4']._modules['1']._modules['bn1'].bias += nn.Parameter(new_beta[3][:64])
        model.features._modules['4']._modules['1']._modules['bn2'].weight += nn.Parameter(new_gamma[4][:64])
        model.features._modules['4']._modules['1']._modules['bn2'].bias += nn.Parameter(new_beta[4][:64])

        #layer 3
        model.features._modules['5']._modules['0']._modules['bn1'].weight += nn.Parameter(new_gamma[5][:128])
        model.features._modules['5']._modules['0']._modules['bn1'].bias += nn.Parameter(new_beta[5][:128])
        model.features._modules['5']._modules['0']._modules['bn2'].weight += nn.Parameter(new_gamma[6][:128])
        model.features._modules['5']._modules['0']._modules['bn2'].bias += nn.Parameter(new_beta[6][:128])

        model.features._modules['5']._modules['1']._modules['bn1'].weight += nn.Parameter(new_gamma[7][:128])
        model.features._modules['5']._modules['1']._modules['bn1'].bias += nn.Parameter(new_beta[7][:128])
        model.features._modules['5']._modules['1']._modules['bn2'].weight += nn.Parameter(new_gamma[8][:128])
        model.features._modules['5']._modules['1']._modules['bn2'].bias += nn.Parameter(new_beta[8][:128])

        #layer 4
        model.features._modules['6']._modules['0']._modules['bn1'].weight += nn.Parameter(new_gamma[9][:256])
        model.features._modules['6']._modules['0']._modules['bn1'].bias += nn.Parameter(new_beta[9][:256])
        model.features._modules['6']._modules['0']._modules['bn2'].weight += nn.Parameter(new_gamma[10][:256])
        model.features._modules['6']._modules['0']._modules['bn2'].bias += nn.Parameter(new_beta[10][:256])

        model.features._modules['6']._modules['1']._modules['bn1'].weight += nn.Parameter(new_gamma[11][:256])
        model.features._modules['6']._modules['1']._modules['bn1'].bias += nn.Parameter(new_beta[11][:256])
        model.features._modules['6']._modules['1']._modules['bn2'].weight += nn.Parameter(new_gamma[12][:256])
        model.features._modules['6']._modules['1']._modules['bn2'].bias += nn.Parameter(new_beta[12][:256])


    
        

        # #extract weights gradients are to be kept, hence first calculate the summation with gradients 
        # weights = []
        # bias = []
        # weights.append(model.features._modules['7']._modules['0']._modules['bn1'].weight)
        # bias.append(model.features._modules['7']._modules['0']._modules['bn1'].bias)
        # weights.append(model.features._modules['7']._modules['0']._modules['bn2'].weight)
        # bias.append(model.features._modules['7']._modules['0']._modules['bn2'].bias)

        # weights.append(model.features._modules['7']._modules['1']._modules['bn1'].weight)
        # bias.append(model.features._modules['7']._modules['1']._modules['bn1'].bias)
        # weights.append(model.features._modules['7']._modules['1']._modules['bn2'].weight)
        # bias.append(model.features._modules['7']._modules['1']._modules['bn2'].bias)
        

        


    return model, added_weights, added_bias





def gradients_extract(d1):
    weights = []

    # detatch first
    d1 = {k: v.detach() for k, v in d1.items()}

    all_weights = []
    all_bias = []

    for k, v in d1.items():
        # check if bn and weight is present
        #check if downsample is not present
        if '7' in k:
            if 'downsample' not in k:
                if 'weight' in k:
                    all_weights.append(v)
                    # print('Key and value are: ', k, v)
                    # print('Key is : ', k)
                if 'bias' in k:
                    all_bias.append(v)
                    # print('Key is : ', k)
                    # print('Key and value are: ', k, v)

    

    # import pdb; pdb.set_trace()
            


    # all_weight_gradients = all_weights
    # target_size = 512

    # for i, tensor in enumerate(all_weight_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size

    #     if pad_size > 0:
    #         all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

    # all_bias_gradients = all_bias

    # for i, tensor in enumerate(all_bias_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size

    #     if pad_size > 0:
    #         all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

    # weights = torch.stack(all_weight_gradients, 0).to(device)
    # bias = torch.stack(all_bias_gradients, 0).to(device)
    all_weight_gradients = torch.stack(all_weights).to(device)
    
    all_bias_gradients = torch.stack(all_bias).to(device)

    
    # import pdb; pdb.set_trace()



    # return weights, bias
    return all_weight_gradients, all_bias_gradients


#gradients extracted from the model for different layers 


def gradients_extract_gen2(d1):
    weights = []

    # detatch first
    d1 = {k: v.detach() for k, v in d1.items()}

    all_weights = []
    all_bias = []

    for k, v in d1.items():
        # check if bn and weight is present
        #check if downsample is not present
        if not '7' in k:
            if 'downsample' not in k:
                if 'weight' in k:
                    all_weights.append(v)
                    # print('Key and value are: ', k, v)
                    # print('Key is : ', k)
                if 'bias' in k:
                    all_bias.append(v)
                    # print('Key is : ', k)
                    # print('Key and value are: ', k, v)

    

    # import pdb; pdb.set_trace()
            


    all_weight_gradients = all_weights
    target_size = 512

    for i, tensor in enumerate(all_weight_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

    all_bias_gradients = all_bias

    for i, tensor in enumerate(all_bias_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

    weights = torch.stack(all_weight_gradients, 0).to(device)
    bias = torch.stack(all_bias_gradients, 0).to(device)
    # all_weight_gradients = torch.stack(all_weights).to(device)
    
    # all_bias_gradients = torch.stack(all_bias).to(device)

    
    # import pdb; pdb.set_trace()



    # return weights, bias
    # return all_weight_gradients, all_bias_gradients
    return weights, bias



def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_tent_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_tent_params(net_copy):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in net_copy.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    # print('names', names)
    # print('params', params)
    # import pdb; pdb.set_trace()
    return params, names


def train_meta_target(epoch, domain_id, scores):
    #model generation in this loop , use only tr_net here
    # net.train() 
    #clone the net it and use it only for this loop 
    #pytorch clon
    net.train()
    tr_net_fc.train()
    # tr_net.train()
    # tr_net_gen2.train()
    #create net copy after tent 
    del scores 
    #delete this model after the loop
    #call gradintes_extract 
    #using scores from previous meta train loop as of now for initialization 
    
    
    # tr_net.train()

    
    t0 = time.time()
    
    loss_model_gen = 0
    # if epoch<3:
    #     domain_id = epoch
    #     loss_rate = 1e-8
    # else:
    # domain_id = np.random.randint(len(domains))
    #generate random domain id
    log_string('\n ')
    log_string('Domain ID %d' % domain_id)
    
    rt_context.reset('train', domain_id, transform=transform_train)

    
    context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_num, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)

    

    #start tent here and collect gradients for rt_context dataloader 
    #use source model and track gradients 
    softmax_loss = 0
    #create new model 
    net_copy = copy.deepcopy(net)
    net_copy.to(device)
    net_copy.train()
    #send to tent to configure model 
    # net_copy = configure_tent_model(net_copy)
    # net_copy.train()
    # net.eval()
    #create a optimizer based on the params to update
    # tent_params, tent_names = collect_tent_params(net_copy)
    #using optimizer only for net_copy not met original 
    # optimizer_tent = torch.optim.AdamW(tent_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    optimizer_meta_target = torch.optim.AdamW(net_copy.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    #net original is the source model
    net_original = copy.deepcopy(net)
    net_original.to(device)
    net_original.train()
    # net_original = configure_tent_model(net_original)
    # tent_params_original, tent_names_original = collect_tent_params(net_original)
    
    #prevous optimizer 
    # optimizer = torch.optim.AdamW([{'params': net.features.parameters(), 'lr':args.lr * res_lr},   # different lr)
    #                           {'params': net.fc.parameters(), 'lr':args.lr}], weight_decay=WEIGHT_DECAY)
    
    train_loss = 0
    correct = 0
    total = 0
    #do cross entropy instead 
    for batch_idx, (inputs, targets, _ ) in enumerate(context_loader):
        inputs , targets = inputs.to(device), targets.to(device)
        optimizer_meta_target.zero_grad()
        outputs, _ = net_copy(inputs)
        #calculate softmax loss
        # loss_sfm = softmax_entropy(outputs).mean(0)
        loss_sfm = F.cross_entropy(outputs, targets)
        softmax_loss += loss_sfm.item()
        loss_sfm.backward()
        # loss_ce.backward
        #calculate accuracy 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #calculate gradients for all layers
        scores = {}
        bn_layer_list = []
        for name, m in net_copy.named_modules():
        #     if isinstance(m, GraphConv):
            # if isinstance(m, torch.nn.Conv2d):
            #     scores[name] = torch.clone(m.weight.grad.clone()).detach()
            # if isinstance(m, torch.nn.BatchNorm2d):
            #for fc layer 
            if isinstance(m, torch.nn.Linear):
                #take both gradients with weight and bias as keys 
                # # scores[name] = torch.clone(m.weight.grad.clone()).detach()
                # print('Name is : ', name)
                # print('Shape of weight is : ', m)
                scores[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
                # scores[name + '.bias'] = torch.clone(m.bias.grad.clone()).detach()
                bn_layer_list.append(name)
        
        
        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        # import pdb; pdb.set_trace()
        grad_flow = torch.norm(all_scores)
        optimizer_meta_target.step()
        # import pdb; pdb.set_trace()
        
        
    log_string('Softmax loss: %f' % (softmax_loss/(batch_idx+1)))
    log_string('Accuracy through entropy minimization: %f (%d/%d)' % (100.*correct/total, correct, total))

    #format is wandb.log({'train/loss': train_loss/(batch_idx+1)})

    wandb.log({'meta_target_acc/softmax_loss': softmax_loss/(batch_idx+1), 'epoch': epoch})
    wandb.log({'meta_target_acc/acc': 100.*correct/total, 'epoch': epoch})

    grad_update_wandb = {}
    # for key, value in scores.items():
    #     #compute norm of the value and update
    #     new_value = torch.norm(value)
    #     # import pdb; pdb.set_trace()
    #     grad_update_wandb[key] = new_value
    #     key = str(key)
    #     wandb.log({'layer_wise/' + key: new_value, 'epoch': epoch})
    #from grad_update_wandb dict, print the key with max value
    # max_key = max(grad_update_wandb, key=grad_update_wandb.get)
    # log_string('Max grad flow layer: %s' % max_key)

    #meta_target 

    #Tent gradients collected 

    

    loss_softmax_ent = 0
    total = 0
    correct = 0
    train_loss = 0
    loss_numerical_tot = 0
    total_source_model = 0
    correct_source_model = 0

    for batch_idx, (inputs, targets, img_name2 ) in enumerate(context_loader):
        #test 
        #loop runs for 1 batch exactly, because of less number of CTX
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        optimizer_tr_net_fc.zero_grad()
        # print('Batch number is ---------------------------------------- ', batch_idx)
        #chnage net_copy to source trained model.
       
        _, outputs_central = net(inputs)
        
        # weight, bias = bn_extractor_all(net)
        weight = fc_extractor_all(net)
        # import pdb; pdb.set_trace()
        
        # weights_bias = torch.cat((weight, bias), dim=0).to(device)
        #size of gradients  is 40, 512
        all_gradients = gradients_extract_fc(scores)
        # import pdb; pdb.set_trace()
        # all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)
        # import pdb; pdb.set_trace()
        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()
        #compute mean of outputs_central 
        outputs_central_mean = outputs_central.mean(0,keepdim=True)
        del outputs_central
        #add an extra dimension to the outputs_central_mean
        # outputs_central_mean = outputs_central_mean.unsqueeze(0)
        #repeat it 40 times 
        outputs_central_mean = outputs_central_mean.repeat(7,1)
       
        # all_inputs = torch.cat((outputs_central_mean, all_gradients), dim=0).to(device)
        # all_inputs = all_inputs.detach()
        # import pdb; pdb.set_trace()
        #CF, G, W 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()d1
        # import pdb; pdb.set_trace()
        outputs_tr_net = tr_net_fc(outputs_central_mean, all_gradients, weight)
        # all_gamma_beta_output = outputs_tr_net
        # import pdb; pdb.set_trace()
        # new_gamma = outputs_tr_net[:4]
        # new_beta = outputs_tr_net[4:]
        # new_gb = torch.cat((new_gamma, new_beta), dim=0).to(device)
        
        # net_copy = weights_generator(net_copy, new_gamma, new_beta) #send source model instead because tent is already optimized
        net_original, added_weights = weights_generator_fc(net_original, outputs_tr_net)
        # added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(device)

        #take tent weights_bias 
        # tent_weights, tent_bias = bn_extractor_all(net_copy)
        # tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(device)

        
        # outputs, _ = net_copy(inputs)
        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent+= loss_softmax.item()
        
        train_loss += loss_tr_net.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #try source model
        outputs_source_model, _ = net(inputs)
        _, predicted_source_model = outputs_source_model.max(1)
        total_source_model += targets.size(0)
        correct_source_model += predicted_source_model.eq(targets).sum().item()



        #use directly full tensor instead of seperately 
        # gamma_beta= torch.cat((new_gamma, new_beta), dim=0)
       
        # loss_numerical = criterion_numerical(gamma_beta, weights_bias)
        #loss numerical between added weights and tent weights
        # print('Loss between added weights and generated weights', criterion_numerical(added_weights_bias, new_gb))
        # loss_numerical = criterion_numerical(added_weights_bias, tent_weights_bias)
        loss_numerical = criterion_numerical(outputs_tr_net, added_weights)
        loss_numerical_tot += loss_numerical.item()
        # loss_numerical_mean = criterion_numerical(gamma_beta.mean(), weights_bias.mean())
        # loss_numerical_tot += loss_numerical_mean.item()
        
        total_loss =   loss_tr_net + loss_numerical
        
        total_loss.backward()
        optimizer_tr_net_fc.step() 



    #log_string
    
    # log_string('Loss model gen: %.3f' % (loss_model_gen/(batch_idx+1)))
    # log_string('Loss tr net: %.3f' % (loss_numerical_tot/(batch_idx+1)))
    # log_string('Loss Cross Entropy: %.3f' % (train_loss/(batch_idx+1)))
    # log_string('Loss Softmax Entropy: %.3f' % (loss_softmax_ent/(batch_idx+1)))
    # log_string('\t Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    log_string('Accuracy of Model Generated: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    log_string('Accuracy of Source Model: %.3f%% (%d/%d)' % (100.*correct_source_model/total_source_model, correct_source_model, total_source_model))
    log_string('Loss Numerical: %.3f' % (loss_numerical_tot/(batch_idx+1)))
    log_string('Epoch: %d, Loss: %.3f' % (epoch, train_loss/(batch_idx+1)))
    
    #writer 
    wandb.log({'train/loss': train_loss/(batch_idx+1)})
    wandb.log({'epoch':epoch})
    wandb.log({'train/acc': 100.*correct/total})
    wandb.log({'train/lr': optimizer.param_groups[0]['lr']})
    wandb.log({'meta_target/tr_net_loss': loss_model_gen/(batch_idx+1)})
    #  wandb.define_metric("meta_target/tr_net_loss", step_metric="epoch")
    wandb.log({'meta_target_acc/source_acc ': 100.*correct_source_model/total_source_model})
    wandb.log({'meta_target_acc/model_gen_acc': 100.*correct/total})
    wandb.log({'meta_target_acc/source_loss': train_loss/(batch_idx+1)})
    wandb.log({'meta_target_acc/model_gen_loss': loss_numerical_tot/(batch_idx+1)})
    wandb.log({'epoch':epoch})

    





    wandb.log({'meta_target/loss': train_loss/(batch_idx+1)})
    wandb.log({'meta_target/acc': 100.*correct/total})
    wandb.log({'meta_target/lr': optimizer.param_groups[0]['lr']})
    # wandb.log({'meta_target/tr_net_loss': loss_model_gen/(batch_idx+1)})
    # wandb.log({'meta_target/grad_flow': grad_flow})
    #start loop for 2nd TR NET 
    # del weights_bias, weights, bias, gamma, beta, gamma_beta, new_gamma, new_beta, new_gb, loss_tr_net, loss_softmax, loss_numerical, total_loss, outputs, predicted, outputs_source_model, predicted_source_model
    '''loss_softmax_ent = 0
    total = 0
    correct = 0
    train_loss = 0
    loss_numerical_tot = 0
    total_source_model = 0
    correct_source_model = 0
    #in this loop use net_original and generate weights using it 
    tr_net_gen2.train()

    for batch_idx, (inputs, targets, img_name2 ) in enumerate(context_loader):
        #test 
        #loop 2 
        #loop runs for 1 batch exactly, because of less number of CTX
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        # optimizer_tr_net.zero_grad()
        optimizer_tr_net_gen2.zero_grad()
        # print('Batch number is ---------------------------------------- ', batch_idx)
        #chnage net_copy to source trained model.
       
        # _, outputs_central = net(inputs)
        # _, outputs_central = net_original(inputs)
        _, outputs_central = net(inputs)
        # import pdb; pdb.set_trace()


        weight, bias = bn_extractor_gen2(net_original)
        # import pdb; pdb.set_trace()
        
        weights_bias = torch.cat((weight, bias), dim=0).to(device)
        #size of gradients  is 40, 512
        all_weight, all_bias = gradients_extract_gen2(scores)
        # import pdb; pdb.set_trace()
        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)
        # import pdb; pdb.set_trace()
        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()
        #compute mean of outputs_central 
        outputs_central_mean = outputs_central.mean(0,keepdim=True)
        del outputs_central
        #add an extra dimension to the outputs_central_mean
        # outputs_central_mean = outputs_central_mean.unsqueeze(0)
        #repeat it 40 times 
        outputs_central_mean = outputs_central_mean.repeat(26,1)
        # import pdb; pdb.set_trace()

       
        # all_inputs = torch.cat((outputs_central_mean, all_gradients), dim=0).to(device)
        # all_inputs = all_inputs.detach()
        # import pdb; pdb.set_trace()
        #CF, G, W 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()d1


        # outputs_tr_net = tr_net(outputs_central_mean, all_gradients, weights_bias)
        outputs_tr_net_gen2 = tr_net_gen2(outputs_central_mean, all_gradients, weights_bias)
        # all_gamma_beta_output = outputs_tr_net
        # import pdb; pdb.set_trace()
        new_gamma = outputs_tr_net_gen2[:13]
        new_beta = outputs_tr_net_gen2[13:]
        new_gb = torch.cat((new_gamma, new_beta), dim=0).to(device)
        
        # net_copy = weights_generator(net_copy, new_gamma, new_beta) #send source model instead because tent is already optimized
        # net_original, added_weights, added_bias = weights_generator(net_original, new_gamma, new_beta)
        net_original, added_weights, added_bias = weights_generator_gen2(net_original, new_gamma, new_beta)
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(device)

        #take tent weights_bias 
        tent_weights, tent_bias = bn_extractor_gen2(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(device)

        
        # outputs, _ = net_copy(inputs)
        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent+= loss_softmax.item()
        
        train_loss += loss_tr_net.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #try source model
        outputs_source_model, _ = net(inputs)
        _, predicted_source_model = outputs_source_model.max(1)
        total_source_model += targets.size(0)
        correct_source_model += predicted_source_model.eq(targets).sum().item()



        #use directly full tensor instead of seperately 
        gamma_beta= torch.cat((new_gamma, new_beta), dim=0)
       
        # loss_numerical = criterion_numerical(gamma_beta, weights_bias)
        #loss numerical between added weights and tent weights
        # print('Loss between added weights and generated weights', criterion_numerical(added_weights_bias, new_gb))
        loss_numerical = criterion_numerical(added_weights_bias, tent_weights_bias)
        loss_numerical_tot += loss_numerical.item()
        # loss_numerical_mean = criterion_numerical(gamma_beta.mean(), weights_bias.mean())
        # loss_numerical_tot += loss_numerical_mean.item()
        
        total_loss =   loss_tr_net + loss_numerical
        
        total_loss.backward()
        # optimizer_tr_net.step() 
        optimizer_tr_net_gen2.step()

    log_string('Accuracy of Model Generated: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    log_string('Accuracy of Source Model: %.3f%% (%d/%d)' % (100.*correct_source_model/total_source_model, correct_source_model, total_source_model))
    log_string('Loss Numerical: %.3f' % (loss_numerical_tot/(batch_idx+1)))
    log_string('Epoch: %d, Loss: %.3f' % (epoch, train_loss/(batch_idx+1)))
    wandb.log({"meta_target_acc/model_gen2_acc": 100.*correct/total})
    wandb.log({"meta_target_acc/model_gen2_loss": loss_numerical_tot/(batch_idx+1)})
    wandb.log({"epoch": epoch})
    # wandb.define_metric("meta_target_acc/model_gen2_acc", step_metric="epoch")
    # wandb.define_metric("meta_target_acc/model_gen2_loss", step_metric="epoch")



    del net_copy, net_original
    tr_net.eval()
    tr_net_gen2.eval()

    print('time elapsed: %f' % (time.time()-t0))'''

NUM_BATCHES_TO_LOG = 2

#to this test function add tent and mimic meta adapt 
def test(epoch):
    #needs to be changed alot 
    global best_acc
    global best_valid_acc
    #log_string('\nEpoch: %d' % epoch)
    # wandb.define_metric("epoch")
    # wandb.define_metric("test/acc", step_metric="epoch")
    # wandb.define_metric("test/loss", step_metric="epoch")
    log_counter = 0 
    #set net to eval mode
    net.eval()
    # tr_net.eval()
    # tr_net_gen2.eval()
    tr_net_fc.eval()
    
    # net.eval()
    #dont touuch the model 
    log_string('-----------------Test----------------- \n')
    

    # tr_net.train() dont use train use eval, we shall see tr_net updation later 
    
    all_dataset.reset('test', 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=cpu_workers, worker_init_fn=worker_init_fn)

    #create tr_net copy
    tr_net_fc_copy = copy.deepcopy(tr_net_fc)
    tr_net_fc_copy.to(device)
    tr_net_fc_copy.train()
    # optimizer_tr_net_copy = torch.optim.AdamW(tr_net_copy.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    


    #start tent loop now .
    net_original = copy.deepcopy(net)
    net_original.to(device)
    net_original.train()

    net_copy = copy.deepcopy(net)
    net_copy.to(device)
    net_copy.train()
    #send to tent to configure model 
    # net_copy = configure_tent_model(net_copy)
    # net_original = configure_tent_model(net_original)
    
    
    # net.eval()
    # #create a optimizer based on the params to update
    # tent_params, tent_names = collect_tent_params(net_copy)

    # optimizer_tent_copy = torch.optim.AdamW(tent_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    #prevous optimizer 
    # optimizer = torch.optim.AdamW([{'params': net.features.parameters(), 'lr':args.lr * res_lr},   # different lr)
    #                           {'params': net.fc.parameters(), 'lr':args.lr}], weight_decay=WEIGHT_DECAY)
    optimizer_normal = torch.optim.AdamW(net_copy.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    total = 0
    correct = 0
    softmax_loss = 0
    optimizer_tr_net_gen2_copy = torch.optim.AdamW(tr_net_fc_copy.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    



    for batch_idx, (inputs, targets, _ ) in enumerate(testloader):
        inputs , targets = inputs.to(device), targets.to(device)
        optimizer_normal.zero_grad()
        #change it to source model 
        outputs, _ = net_copy(inputs)
        #calculate softmax loss
        #generate pseudo label 
        pseudo_label = torch.argmax(outputs, dim=1)
        #detach pseudo label
        pseudo_label = pseudo_label.detach()
        # loss_sfm = softmax_entropy(outputs).mean(0)
        loss_sfm = F.cross_entropy(outputs, pseudo_label)
        softmax_loss += loss_sfm.item()
        loss_sfm.backward()
        #calculate accuracy 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #calculate gradients for all layers
        scores = {}
        bn_layer_list = []
        # for name, m in net_copy.named_modules():
       
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         #take both gradients with weight and bias as keys 
        #         # scores[name] = torch.clone(m.weight.grad.clone()).detach()
        #         scores[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
        #         scores[name + '.bias'] = torch.clone(m.bias.grad.clone()).detach()
        #         bn_layer_list.append(name)
        for name, m in net_copy.named_modules():
        #     if isinstance(m, GraphConv):
            # if isinstance(m, torch.nn.Conv2d):
            #     scores[name] = torch.clone(m.weight.grad.clone()).detach()
            # if isinstance(m, torch.nn.BatchNorm2d):
            #for fc layer 
            if isinstance(m, torch.nn.Linear):
                #take both gradients with weight and bias as keys 
                # # scores[name] = torch.clone(m.weight.grad.clone()).detach()
                # print('Name is : ', name)
                # print('Shape of weight is : ', m)
                scores[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
                # scores[name + '.bias'] = torch.clone(m.bias.grad.clone()).detach()
                bn_layer_list.append(name)
        # import pdb; pdb.set_trace()
        
        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
        # import pdb; pdb.set_trace()
        grad_flow = torch.norm(all_scores)
        optimizer_normal.step()


    log_string('\tSoftmax loss: %f' % (softmax_loss/(batch_idx+1)))
    log_string('\tAccuracy through entropy: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    grad_update_wandb = {}
    # for key, value in scores.items():
    #     #compute norm of the value and update
    #     new_value = torch.norm(value)
    #     # import pdb; pdb.set_trace()
    #     grad_update_wandb[key] = new_value
    #     key = str(key)
    #     wandb.log({'layer_wise/' + key: new_value, 'epoch': epoch})
    # #from grad_update_wandb dict, print the key with max value
    # max_key = max(grad_update_wandb, key=grad_update_wandb.get)
    # log_string('\tMax grad flow layer: %s' % max_key)
    log_string('\n ')

    

    loss_softmax_ent = 0
    total = 0
    correct = 0
    test_loss = 0
    model_gen_numerical = 0

    total_source_model = 0
    correct_source_model = 0
    optimizer_tr_net_copy = torch.optim.AdamW(tr_net_fc_copy.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    


    for batch_idx, (inputs, targets, img_name2 ) in enumerate(testloader):
        #test 
        #loop runs for 1 batch exactly, because of less number of CTX
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        # optimizer_tr_net_copy.zero_grad()
        optimizer_tr_net_copy.zero_grad()

        _, outputs_central = net_original(inputs)
        # weight, bias = bn_extractor_all(net_original)
        # weights_bias = torch.cat((weight, bias), dim=0).to(device)
        weight = fc_extractor_all(net)
        # all_weight, all_bias = gradients_extract(scores)
        all_gradients = gradients_extract_fc(scores)
        # import pdb; pdb.set_trace()
        # all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)
        # import pdb; pdb.set_trace()
        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()
        #compute mean of outputs_central 
        outputs_central_mean = outputs_central.mean(0,keepdim=True)
        del outputs_central
        #add an extra dimension to the outputs_central_mean
        # outputs_central_mean = outputs_central_mean.unsqueeze(0)
        #repeat it 16 times 
        outputs_central_mean = outputs_central_mean.repeat(7,1)
        outputs_tr_net = tr_net_fc_copy(outputs_central_mean, all_gradients, weight)
        # all_gamma_beta_output = outputs_tr_net
        # import pdb; pdb.set_trace()
        
        
        
        # import pdb; pdb.set_trace()
        
        # new_gamma = outputs_tr_net[:4]
        # new_beta = outputs_tr_net[4:]
        # net_copy = weights_generator(net_copy, new_gamma, new_beta)
        # net_original, added_weights, added_bias = weights_generator(net_original, new_gamma, new_beta)
        # added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(device)
        net_original, added_weights = weights_generator_fc(net_original, outputs_tr_net)
        # a

        # #take tent weights_bias 
        # tent_weights, tent_bias = bn_extractor_all(net_copy)
        # tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(device)


        outputs, _ = net_original(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # loss_softmax = softmax_entropy(outputs).mean(0)
        # loss_softmax_ent+= loss_softmax.item()

        # gamma_beta = torch.cat((new_gamma, new_beta), dim=0).to(device)

        # loss_numerical = criterion_numerical(gamma_beta, weights_bias)
        # loss_numerical = criterion_numerical(added_weights_bias, tent_weights_bias)
        loss_numerical = criterion_numerical(outputs_tr_net, added_weights)
        model_gen_numerical += loss_numerical.item()

        total_loss =  loss_numerical
        t_ls = criterion(outputs, targets)
        test_loss += t_ls.item()
        
        
        # total_loss.backward()
        total_loss = loss_numerical
        

        
        
        total_loss.backward()
        
        optimizer_tr_net_copy.step()

    log_string('\tSoftmax loss: %f' % (loss_softmax_ent/(batch_idx+1)))
    #print loss and accuracy
    log_string('\tAccuracy of model gen : %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    log_string('\tTest loss: %f' % (test_loss/(batch_idx+1)))
    log_string('\tModel gen loss numerical : %f' % (model_gen_numerical/(batch_idx+1)))

    wandb.log({'test/model_gen_acc': 100.*correct/total, 'epoch': epoch})
    wandb.log({'test/model_gen_loss': model_gen_numerical/(batch_idx+1), 'epoch': epoch})
    acc_tr_net1 = 100.*correct/total

    #gen loop 2 
    '''
    #delete models 
    # del net_copy, net_original, tr_net_copy
    del tr_net_copy
    #delete unecessary variables
    del inputs, targets, outputs, outputs_central_mean, outputs_tr_net, predicted, gamma_beta, new_gamma, new_beta, all_gradients, weights_bias, all_weight, all_bias, weight, bias, tent_weights, tent_bias, tent_weights_bias, added_weights, added_bias, added_weights_bias

    tr_net_gen2_copy = copy.deepcopy(tr_net_gen2)
    tr_net_gen2_copy.to(device)
    tr_net_gen2_copy.train()

    loss_softmax_ent = 0
    total = 0
    correct = 0
    test_loss = 0
    model_gen_numerical = 0

    total_source_model = 0
    correct_source_model = 0
    loss_numerical_tot = 0

    for batch_idx, (inputs, targets, img_name2 ) in enumerate(testloader):
        #test 
        #loop 2 
        #loop runs for 1 batch exactly, because of less number of CTX
        inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        # optimizer_tr_net.zero_grad()
        # optimizer_tr_net_gen2.zero_grad()
        # print('Batch number is ---------------------------------------- ', batch_idx)
        #chnage net_copy to source trained model.
       
        # _, outputs_central = net(inputs)
        _, outputs_central = net_original(inputs)
        # import pdb; pdb.set_trace()


        weight, bias = bn_extractor_gen2(net_original)
        # import pdb; pdb.set_trace()
        
        weights_bias = torch.cat((weight, bias), dim=0).to(device)
        #size of gradients  is 40, 512
        all_weight, all_bias = gradients_extract_gen2(scores)
        # import pdb; pdb.set_trace()
        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)
        # import pdb; pdb.set_trace()
        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()
        
        #compute mean of outputs_central 
        outputs_central_mean = outputs_central.mean(0,keepdim=True)
        del outputs_central
        #add an extra dimension to the outputs_central_mean
        # outputs_central_mean = outputs_central_mean.unsqueeze(0)
        #repeat it 40 times 
        outputs_central_mean = outputs_central_mean.repeat(26,1)
        # import pdb; pdb.set_trace()

       
        # all_inputs = torch.cat((outputs_central_mean, all_gradients), dim=0).to(device)
        # all_inputs = all_inputs.detach()
        # import pdb; pdb.set_trace()
        #CF, G, W 
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()d1


        # outputs_tr_net = tr_net(outputs_central_mean, all_gradients, weights_bias)
        outputs_tr_net = tr_net_gen2_copy(outputs_central_mean, all_gradients, weights_bias)
        # all_gamma_beta_output = outputs_tr_net
        # import pdb; pdb.set_trace()
        new_gamma = outputs_tr_net[:13]
        new_beta = outputs_tr_net[13:]
        # new_gb = torch.cat((new_gamma, new_beta), dim=0).to(device)
        
        # net_copy = weights_generator(net_copy, new_gamma, new_beta) #send source model instead because tent is already optimized
        # net_original, added_weights, added_bias = weights_generator(net_original, new_gamma, new_beta)
        net_original, added_weights, added_bias = weights_generator_gen2(net_original, new_gamma, new_beta)
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(device)

        #take tent weights_bias 
        tent_weights, tent_bias = bn_extractor_gen2(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(device)

        
        # outputs, _ = net_copy(inputs)
        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent+= loss_softmax.item()
        
        test_loss += loss_tr_net.item()
        
        _, predicted = outputs.max(1)
        del outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # #try source model
        # outputs_source_model, _ = net(inputs)
        # _, predicted_source_model = outputs_source_model.max(1)
        # total_source_model += targets.size(0)
        # correct_source_model += predicted_source_model.eq(targets).sum().item()



        #use directly full tensor instead of seperately 
        gamma_beta= torch.cat((new_gamma, new_beta), dim=0)
       
        # loss_numerical = criterion_numerical(gamma_beta, weights_bias)
        #loss numerical between added weights and tent weights
        # print('Loss between added weights and generated weights', criterion_numerical(added_weights_bias, new_gb))
        loss_numerical = criterion_numerical(added_weights_bias, tent_weights_bias)
        loss_numerical_tot += loss_numerical.item()
        #why is the loss zero 
        # import pdb; pdb.set_trace()

        # loss_numerical_mean = criterion_numerical(gamma_beta.mean(), weights_bias.mean())
        # loss_numerical_tot += loss_numerical_mean.item()
        
        # total_loss =   loss_tr_net + loss_numerical

    log_string('\tSoftmax loss: %f' % (loss_softmax_ent/(batch_idx+1)))
    #print loss and accuracy
    log_string('\tAccuracy of model gen : %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    log_string('\tTest loss: %f' % (test_loss/(batch_idx+1)))
    log_string('\tModel gen loss numerical : %f' % (loss_numerical_tot/(batch_idx+1)))


    #wandb 
    wandb.define_metric("test/model_gen2_acc", step_metric="epoch")
    wandb.define_metric("test/model_gen2_loss", step_metric="epoch")
    wandb.log({"test/model_gen2_acc": 100.*correct/total, "test/model_gen2_loss": loss_numerical_tot/(batch_idx+1)})
    wandb.log({"epoch": epoch})


    acc_tr_net2 = 100.*correct/total'''


        





    #model saving weights 
    # acc = 100.*correct/total
    # source_acc = 100.*correct_source_model/total_source_model
    # if acc > best_valid_acc:
    acc = acc_tr_net1
    if acc > best_valid_acc:
        best_valid_acc = acc
        
        print('Saving best model... %f' % acc)
        state = {
            'net': net.state_dict(),
            'tr_net_fc': tr_net_fc.state_dict(),
            'model_gen_acc': acc,
            'epoch': epoch,
        }
        #save in log_dir 
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth'))
        wandb.summary['model_path'] = os.path.join(checkpoint_dir, 'best_model.pth')
        wandb.summary['model_path_epoch'] = epoch


    # del net_copy, optimizer_tent_copy, net_original

    












 #normal test function 
def normal_test(epoch):
    global best_acc
    #log_string('\nEpoch: %d' % epoch)
    # wandb.define_metric("epoch")
    # wandb.define_metric("test/acc", step_metric="epoch")
    # wandb.define_metric("test/loss", step_metric="epoch")
    log_counter = 0 
    
    net.eval()
    tr_net_fc.eval()
    # tr_net.train()
    all_dataset.reset('test', 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=cpu_workers, worker_init_fn=worker_init_fn)
    # rt_context.reset('test', 0, transform=transform_test)
    # context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=cpu_workers, drop_last=False, worker_init_fn=worker_init_fn)
    # for batch_idx, (inputs, targets,  img_name1 ) in enumerate(context_loader):
    #     context_img, context_label = inputs.to(device), targets.to(device)
    test_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    batch_count = 0 
    '''with torch.no_grad():

        for batch_idx, (inputs, targets,  img_name1 ) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _= net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()'''
    #skip default loop, instead predict batch norm every time now 
    with torch.no_grad():

        for batch_idx, (inputs, targets,  img_name1 ) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # _, outputs_central = net(inputs)
            # #start logic 
            # # weights_from_model = net.features._modules['7']._modules['0']._modules['bn1'].weight
            # # bias_from_model = net.features._modules['7']._modules['0']._modules['bn1'].bias
            # weights_from_model, bias_from_model = bn_extractor_eval(copy.deepcopy(net))
            # weights_bias = torch.vstack ((weights_from_model, bias_from_model))
            # outputs_tr_net = tr_net(weights_bias, outputs_central)
            # new_gamma, new_beta = outputs_tr_net

            # #generate new batch norm weights
            # weights_generator(net, new_gamma, new_beta)
            
            #mean of samples 
        

            # net.features._modules['7']._modules['0']._modules['bn1'].weight = torch.nn.Parameter(new_gamma)
            # net.features._modules['7']._modules['0']._modules['bn1'].bias = torch.nn.Parameter(new_beta)

            # del outputs 
            #get outputs again with new batch norm weights calculated by the net
            outputs, _ = net(inputs)
            #do the same procedure 

           


            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            


        #send log of loss per epoch to logstring
        log_string('\t Test Loss %f, Acc: %f' % (test_loss/(batch_idx+1), 100.*correct/total))
        

        #writer 
        # te_writer.add_scalar('te/loss',  test_loss/batch_idx+1, epoch)
        # te_writer.add_scalar('te/acc', 100.*correct/total, epoch)
        wandb.log({'normal_test/acc': 100.*correct/total})
        wandb.log({'normal_test/loss': test_loss/batch_idx+1})
        wandb.log({'epoch':epoch})
        

    '''acc = 100.*correct/total
    if acc > best_valid_acc:
        print('Saving best model... %f' % acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #save in log_dir 
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth'))'''
        
        
               
        

def validation(epoch):
    global best_valid_acc
    wandb.define_metric("epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/acc", step_metric="epoch")
    
    net.eval()
    tr_net.eval()

    test_loss = 0
    correct = 0
    total = 0
    ac_correct = [0, 0, 0]
        
    # all_dataset.reset('val', 0, transform=transform_test)
    # valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    # rt_context.reset('val', transform=transform_test)
    # context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    
    
    with torch.no_grad():
        for i in range(4):
            all_dataset.reset('val', i, transform=transform_test)
            valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

            num_preds = 1

            for batch_idx, (inputs, targets, _) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs, _= net(inputs)
                loss = criterion(outputs, targets)


                # ys, _, _, _ = net(inputs, context_img, i, ctx_test, ifsample)

                # ys = ys.view(ys.size()[0], -1, args.num_classes)

                # y = torch.softmax(ys, -1).mean(1)
                
                # cls_loss = criterion(y, targets)
                # loss = cls_loss

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
    log_string('VAL Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # writer.add_scalar('val_loss', test_loss/(batch_idx+1), epoch)
    # writer.add_scalar('val_acc', 100.*correct/total, epoch)
    # # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_valid_acc:
    #     print('Saving..')
    #     log_string('The best validation Acc')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
    #     best_valid_acc = acc
    # val_loss = 0
    # correct = 0
    # total = 0
    # t0 = time.time()
    # rt_context.reset('val', domain_id = 0, transform=transform_test)
    # context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=cpu_workers, drop_last=False, worker_init_fn=worker_init_fn)
    # for batch_idx, (inputs, targets,  img_name1 ) in enumerate(context_loader):
    #     context_img, context_label = inputs.to(device), targets.to(device)

    # with torch.no_grad():
    #     for i in range(4):
    #         all_dataset.reset('val', i, transform=transform_test)
    #         valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=cpu_workers, worker_init_fn=worker_init_fn)

            

    #         for batch_idx, (inputs, targets,  img_name1) in enumerate(valloader):
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             optimizer.zero_grad()
    #             outputs, _ = net(inputs)
    #             loss = criterion(outputs, targets)
    #             val_loss += loss.item()
    #             predicted, _ = outputs.max(1)
    #             total += targets.size(0)
    #             correct += predicted.eq(targets).sum().item()
    #     log_string('\t \t Val Loss: %f, Acc: %f' % (val_loss/(batch_idx+1), 100.*correct/total))
    #     #writer
    #     val_writer.add_scalar('val/loss', val_loss/(batch_idx+1), epoch)
    #     val_writer.add_scalar('val/acc', 100.*correct/total, epoch)
    #     #wandb
    wandb.log({'val/acc': 100.*correct/total})
    wandb.log({'val/loss': test_loss/(batch_idx+1)})
    wandb.log({'epoch':epoch})

    # #save checkpoint
    # acc = 100.*correct/total
    # if acc > best_valid_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     #save in log_dir 
    #     checkpoint_dir = os.path.join(LOG_DIR, 'checkpoint')
    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)
    #     torch.save(state, os.path.join(checkpoint_dir, 'best_val.pth'))
    #     # #save optimizer also 
    #     # torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'best_val_optim.pth'))
    #     best_valid_acc = acc
    
    return 0





decay_ite = [0.6*max_ite]

if args.autodecay:
    for epoch in range(300):
        train(epoch)
        f = test(epoch)
        if f == 0:
            converge_count = 0
        else:
            converge_count += 1

        if converge_count == 20:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.2
            log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            converge_count = 0

        if optimizer.param_groups[0]['lr'] < 2e-6:
            exit()

else:
    if not iteration_training:
        for epoch in range(start_epoch, 10000000000000):
            if epoch in decay_inter:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
                log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            train(epoch)
            #call for pseudo target 
            train_meta_target(epoch)
            if epoch % 5 == 0:
                _ = validation(epoch)
                _ = test(epoch)
                normal_test(epoch)
    else:
        for epoch in range(10000000000):   
            if epoch in decay_ite:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr']*0.1
                log_string('In iteration %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            domain_id, scores =  train(epoch)
            train_meta_target(epoch, domain_id, scores)
            
            # if epoch % test_ite == 0:
            #     if args.dataset!='office':
            #         _ = validation(epoch)
            #     _ = test(epoch)
            # if epoch % 5 == 0:
            #     _ = validation(epoch)
            #     _ = test(epoch) #skip test for now 
            #     normal_test(epoch)

            if epoch > 200 and epoch % 5 == 0:
                # _ = validation(epoch)
                _ = test(epoch) #skip test for now 
                normal_test(epoch)

            if epoch % 250 == 0:
                _ = test(epoch) #skip test for now 
                normal_test(epoch)



#activation checker 

# # Compute the average activation of each layer
#             for name, module in model.named_modules():
#                 if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#                     if name not in layer_intensity:
#                         layer_intensity[name] = torch.mean(module.weight.abs()).item()
#                     else:
#                         layer_intensity[name] += torch.mean(module.weight.abs()).item()