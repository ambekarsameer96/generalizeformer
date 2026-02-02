from __future__ import print_function

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

from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from aug import *
import pdb
from pacs_rtdataset_new import *
from pacs_dataset import *


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

import transformer_model_confg_67
import dg_model_layer67

import math

import pdb

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument(
    "--weight_decay", default=5e-4, type=float, help="learning rate"
)
parser.add_argument("--sparse", default=0, type=float, help="L1 panelty")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--gpu", default="0", help="GPU to use [default: GPU 0]")
parser.add_argument("--log_dir", default="log1", help="Log dir [default: log]")
parser.add_argument("--dataset", default="PACS", help="datasets")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch Size during training [default: 32]",
)
parser.add_argument(
    "--pseudo_label_update_epoch",
    default=10,
    type=int,
    help="epoch to update pseudo labels",
)

parser.add_argument(
    "--shuffle",
    type=int,
    default=0,
    help="Batch Size during training [default: 32]",
)
parser.add_argument(
    "--optimizer", default="adamw", help="AdamW or momentum [default: AdamW]"
)

parser.add_argument("--net", default="res18", help="res18 or res50")


parser.add_argument("--autodecay", action="store_true")


parser.add_argument(
    "--test_domain", default="art_painting", help="GPU to use [default: GPU 0]"
)
parser.add_argument(
    "--train_domain", default="", help="GPU to use [default: GPU 0]"
)
parser.add_argument(
    "--ite_train", default=True, type=bool, help="learning rate"
)
parser.add_argument("--max_ite", default=10000, type=int, help="max_ite")
parser.add_argument("--test_ite", default=50, type=int, help="learning rate")
parser.add_argument("--bias", default=1, type=int, help="whether sample")
parser.add_argument(
    "--test_batch", default=100, type=int, help="learning rate"
)
parser.add_argument("--data_aug", default=1, type=int, help="whether sample")
parser.add_argument("--difflr", default=1, type=int, help="whether sample")


parser.add_argument(
    "--reslr", default=0.5, type=float, help="backbone learning rate"
)

parser.add_argument(
    "--agg_model", default="concat", help="concat or bayes or rank1"
)
parser.add_argument(
    "--agg_method", default="mean", help="ensemble or mean or ronly"
)


parser.add_argument("--ctx_num", default=10, type=int, help="learning rate")
parser.add_argument("--hierar", default=2, type=int, help="hierarchical model")


parser.add_argument(
    "--model_saving_dir",
    default="./models_new/models_code",
    type=str,
    help=" place to save the best model obtained during training",
)

parser.add_argument(
    "--resume_from_checkpoint", type=str, help=" resume from checkpoint"
)

parser.add_argument(
    "--tr_net_layers", type=int, default=8, help="number of tr_net layers"
)

args = parser.parse_args()
wandb.config.update(args)

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net

max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain


ctx_num = args.ctx_num


difflr = args.difflr
res_lr = args.reslr
hierar = args.hierar
agg_model = args.agg_model


with_bias = args.bias
with_bias = bool(with_bias)
difflr = bool(difflr)
pseudo_label_update_epoch = args.pseudo_label_update_epoch


data_aug = args.data_aug
data_aug = bool(data_aug)
model_saving_dir = args.model_saving_dir
resume_from_checkpoint = args.resume_from_checkpoint
tr_net_layers = args.tr_net_layers


wandb_run_name = wandb.run.name


LOG_DIR = os.path.join("logs_meta", wandb_run_name)
MODEL_DIR = os.path.join(model_saving_dir, wandb_run_name)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(os.path.join(LOG_DIR, "validation")):
    os.makedirs(os.path.join(LOG_DIR, "validation"))

if not os.path.exists(os.path.join(LOG_DIR, "test")):
    os.makedirs(os.path.join(LOG_DIR, "test"))


if not os.path.exists(os.path.join(LOG_DIR, "logs")):
    os.makedirs(os.path.join(LOG_DIR, "logs"))
text_file = os.path.join(LOG_DIR, "log_train.txt")
text_file2 = os.path.join(LOG_DIR, "log_std_output.txt")


import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(text_file2, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):

        pass


sys.stdout = Logger()

LOG_FOUT = open(text_file, "w")

print(args)
LOG_FOUT.write(str(args) + "\n")


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    if print_out:
        print(out_str)


log_string("WANB Project name", wandb_run_name)
log_string("Saving models to ", MODEL_DIR)

log_string("==> Writing text file and stdout pushing file output to ")
log_string(text_file)
log_string(text_file2)


tr_writer = SummaryWriter(LOG_DIR)
val_writer = SummaryWriter(os.path.join(LOG_DIR, "validation"))
te_writer = SummaryWriter(os.path.join(LOG_DIR, "test"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.tensorboard.patch(root_logdir=LOG_DIR)

cpu_workers = 4


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


best_acc = 0
best_valid_acc = 0
start_epoch = 0


decay_inter = [250, 450]


print("==> Preparing data..")

if args.dataset == "PACS":
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0

    ctx_test = ctx_num
    domains = ["art_painting", "photo", "cartoon", "sketch"]
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(",")
    log_string("data augmentation is " + str(data_aug))
    if data_aug:

        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2
                ),
                transforms.RandomHorizontalFlip(),
                ImageJitter(jitter_param),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    log_string("train_domain: " + str(domains))
    log_string("test: " + str(test_domain))

    all_dataset = PACS(test_domain)
    rt_context = rtPACS(test_domain, ctx_num)
else:
    raise NotImplementedError


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird


print("--> --> LOG_DIR <-- <--", LOG_DIR)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


net = dg_model_layer67.ResNet18()


import transofmer_model_feautres

tr_net = transofmer_model_feautres.TransformerModel_67(
    feature_dim=512, num_class=10, nlayers=tr_net_layers
)
tr_net_gen2 = transofmer_model_feautres.TransformerModel_67(
    feature_dim=512, num_class=10, nlayers=tr_net_layers
)


print("==> Building model..")


net.apply(inplace_relu)


net = net.to(device)
tr_net = tr_net.to(device)
tr_net_gen2 = tr_net_gen2.to(device)


pc = get_parameter_number(net)
log_string(
    "Total: %.4fM, Trainable: %.4fM"
    % (pc["Total"] / float(1e6), pc["Trainable"] / float(1e6))
)

"""old 

bn_count = 0
net.eval()
with torch.no_grad():
        
    for class1,m in net.features.named_modules():
        
        
        
        if isinstance(m, nn.BatchNorm2d):

        
        
            print(m)
            bn_count += 1
            
            
            
            import pdb
            pdb.set_trace()
            bn_name = m[0]

            
            layer_values = m.state_dict()
            print('Before weights are: ', m.state_dict()['weight'])
            new_wt = torch.zeros_like(layer_values['weight'])
            
            
            print('created weight: ', new_wt)
            m.state_dict()['weight'] =  torch.nn.parameter.Parameter(new_wt)
            print('After weights are: ', m.state_dict()['weight'])
            exit()"""


python_file_name = os.path.basename(__file__)

current_directory = os.getcwd()
python_file_name = os.path.join(current_directory, python_file_name)

wandb.save(python_file_name)
log_string("Uploaded file: %s" % python_file_name)


wandb.define_metric("epoch")
wandb.define_metric("meta_target/loss", step_metric="epoch")
wandb.define_metric("meta_target/acc", step_metric="epoch")
wandb.define_metric("meta_target/lr", step_metric="epoch")
wandb.define_metric("meta_target/tr_net_loss", step_metric="epoch")


wandb.define_metric("train/loss", step_metric="epoch")
wandb.define_metric("train/acc", step_metric="epoch")
wandb.define_metric("train/lr", step_metric="epoch")


wandb.define_metric("meta_target_acc/source_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/source_loss", step_metric="epoch")

wandb.define_metric("meta_target_acc/entropy_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/entropy_loss", step_metric="epoch")


wandb.define_metric("meta_target_acc/model_gen_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/model_gen_loss", step_metric="epoch")


wandb.define_metric("meta_target_acc/model_gen2_acc", step_metric="epoch")
wandb.define_metric("meta_target_acc/model_gen2_loss", step_metric="epoch")


wandb.define_metric("test/source_acc", step_metric="epoch")
wandb.define_metric("test/source_loss", step_metric="epoch")

wandb.define_metric("test/entropy_acc", step_metric="epoch")
wandb.define_metric("test/entropy_loss", step_metric="epoch")

wandb.define_metric("test/model_gen_acc", step_metric="epoch")
wandb.define_metric("test/model_gen_loss", step_metric="epoch")


wandb.define_metric("test/model_gen2_acc", step_metric="epoch")
wandb.define_metric("test/model_gen2_loss", step_metric="epoch")


wandb.define_metric("normal_test/acc", step_metric="epoch")
wandb.define_metric("normal_test/loss", step_metric="epoch")


conv_layer_list = [
    "features.0",
    "features.4.0.conv1",
    "features.4.0.conv2",
    "features.4.1.conv1",
    "features.4.1.conv2",
    "features.5.0.conv1",
    "features.5.0.conv2",
    "features.5.0.downsample.0",
    "features.5.1.conv1",
    "features.5.1.conv2",
    "features.6.0.conv1",
    "features.6.0.conv2",
    "features.6.0.downsample.0",
    "features.6.1.conv1",
    "features.6.1.conv2",
    "features.7.0.conv1",
    "features.7.0.conv2",
    "features.7.0.downsample.0",
    "features.7.1.conv1",
    "features.7.1.conv2",
]

for layer in conv_layer_list:
    wandb.define_metric("layer_wise/" + layer, step_metric="epoch")


net.train()
tr_net.train()
wandb.watch(net, log="all", log_freq=10)
wandb.watch(tr_net, log="all", log_freq=10)

if device == "cuda":

    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay


decay_inter_optim = [500, 20000, 30000, 40000]

optimizer = torch.optim.AdamW(
    [
        {"params": net.features.parameters(), "lr": args.lr * res_lr},
        {"params": net.fc.parameters(), "lr": args.lr},
    ],
    weight_decay=WEIGHT_DECAY,
)


optimizer_tr_net = torch.optim.AdamW(
    [{"params": tr_net.parameters(), "lr": 0.01}], weight_decay=WEIGHT_DECAY
)
optimizer_tr_net_gen2 = torch.optim.AdamW(
    [{"params": tr_net_gen2.parameters(), "lr": 0.01}],
    weight_decay=WEIGHT_DECAY,
)

scheduler_tr_net = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_tr_net, milestones=decay_inter_optim, gamma=0.1
)
scheduler_tr_net_gen2 = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_tr_net_gen2, milestones=decay_inter_optim, gamma=0.1
)


print("Using optimizer: %s" % optimizer)

for param_group in optimizer.param_groups:
    print(param_group["lr"])
for param_group in optimizer_tr_net.param_groups:
    print(param_group["lr"])
for param_group in optimizer_tr_net_gen2.param_groups:
    print(param_group["lr"])


if args.resume_from_checkpoint:

    print("==> Resuming from checkpoint..")

    checkpoint = torch.load(resume_from_checkpoint)
    net.load_state_dict(checkpoint["net"])
    tr_net.load_state_dict(checkpoint["tr_net"])
    tr_net_gen2.load_state_dict(checkpoint["tr_net_gen2"])
    best_acc = checkpoint["model_gen_acc"]
    start_epoch = checkpoint["epoch"]
    print("Epoch %d, best_acc %f" % (start_epoch, best_acc))

    print("==> Resuming from checkpoint.. done")


def compute_layerwise_metrics(net, loader):

    net.train()
    layer_names = [n for n, _ in net.named_parameters() if "bn" not in n]

    metrics = defaultdict(list)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y, _ in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits, _ = net(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent,
            inputs=net.parameters(),
            retain_graph=True,
            allow_unused=True,
        )

        xent_grads.append([g for g in grad_xent])

        loss_entropy = (
            torch.distributions.categorical.Categorical(logits=logits)
            .entropy()
            .mean()
        )
        grad_entropy = torch.autograd.grad(
            outputs=loss_entropy,
            inputs=net.parameters(),
            retain_graph=True,
            allow_unused=True,
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
            _metrics["rel_grad_abs"].append(
                (grad.abs() / (param.abs() + 1e-6)).mean().item()
            )

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(net, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[f"xent_{k}"].append(v)
    for entropy_grad in entropy_grads:
        ent_grad_metrics = get_grad_norms(net, entropy_grad)
        for k, v in ent_grad_metrics.items():
            metrics[f"ent_{k}"].append(v)

    num_pointwise = min(10, loader.batch_size)
    pt_xent_grads, pt_ent_grads = [], []
    x, y, _ = next(iter(loader))
    x, y = x.cuda(), y.cuda()

    logits, _ = net(x)
    loss_xent_pointwise = F.cross_entropy(logits, y, reduction="none")[
        :num_pointwise
    ]
    for _loss in loss_xent_pointwise:
        grad_xent_pt = torch.autograd.grad(
            outputs=_loss, inputs=net.parameters(), retain_graph=True
        )
        pt_xent_grads.append([g.detach() for g in grad_xent_pt])

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
            for (name, _), g1, g2 in zip(
                net.named_parameters(), grads1, grads2
            ):
                if name not in layer_names:
                    continue
                cosine_sims.append(
                    F.cosine_similarity(
                        g1.flatten(), g2.flatten(), dim=0
                    ).item()
                )
            all_cosine_sims.append(cosine_sims)
        return all_cosine_sims

    metrics["xent_pairwise_cosine_sim"] = get_pointwise_grad_norms(
        net, pt_xent_grads
    )
    metrics["ent_pairwise_cosine_sim"] = get_pointwise_grad_norms(
        net, pt_ent_grads
    )
    from matplotlib import pyplot as plt


from collections import defaultdict
import itertools


def get_lr_weights(net, loader):

    layer_names = [n for n, _ in net.named_parameters() if "bn" not in n]

    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y, _ in partial_loader:
        x, y = x.cuda(), y.cuda()

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
            if 1 == 2:
                tmp = (grad * grad) / (
                    torch.var(grad, dim=0, keepdim=True) + 1e-8
                )
                _metrics[name] = tmp.mean().item()
            else:
                _metrics[name] = (
                    torch.norm(grad).item() / torch.norm(param).item()
                )

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(net, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics


def train(epoch):

    net.train()

    tr_net.eval()
    tr_net_gen2.eval()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    kl_loss_tot = 0
    w_loss_tot = 0
    js_div_tot = 0
    correct_source = 0
    total_source = 0
    adapt_loss_tot = 0
    if epoch < 3:
        domain_id = epoch
        loss_rate = 1e-8
    else:
        random.seed(6465)
        domain_id = np.random.randint(len(domains))
        loss_rate = 1
    log_string("\n ")
    log_string("Domain ID %d" % domain_id)

    all_dataset.reset("train", domain_id, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cpu_workers,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    """ Commenting SF now since the blocks are already known
    SF says FC is also being updated, certianly because it is from scratch and resnet is pretrained 
    layer_weights = [0 for layer, _ in net.named_parameters() if 'bn' not in layer]
    layer_names = [layer for layer, _ in net.named_parameters() if 'bn' not in layer]
    compute_layerwise_metrics(net, trainloader)
    weights = get_lr_weights(net, trainloader)
    max_weight = max(weights.values())
    for k, v in weights.items(): 
        weights[k] = v / max_weight
    layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
    
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
    

    import pdb; pdb.set_trace()"""

    for batch_idx, (inputs, targets, img_name2) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs, _ = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        scores = {}
        for name, m in net.named_modules():

            if isinstance(m, torch.nn.Conv2d):
                scores[name] = torch.clone(m.weight.grad.clone()).detach()
            if isinstance(m, torch.nn.BatchNorm2d):

                scores[name + "_weight"] = torch.clone(
                    m.weight.grad.clone()
                ).detach()
                scores[name + "_bias"] = torch.clone(
                    m.bias.grad.clone()
                ).detach()

        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])

        grad_flow = torch.norm(all_scores)

        optimizer.step()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if iteration_training and batch_idx >= batchs_per_epoch:
            break

    log_string(
        "Epoch: %d, Loss: %.3f, Acc: %.3f"
        % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
    )

    wandb.log({"train/loss": train_loss / (batch_idx + 1)})
    wandb.log({"epoch": epoch})
    wandb.log({"train/acc": 100.0 * correct / total})
    wandb.log({"train/lr": optimizer.param_groups[0]["lr"]})

    print("time elapsed: %f" % (time.time() - t0))

    return domain_id, scores


criterion_numerical = torch.nn.HuberLoss()


"""
def bn_extractor(net):
    bn_list = []
    weights = []
    bias = []

    
    
    
    


    
    
    
    



    
    
    
    

    
    
    
    
    


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

    
    return weights_bias

"""


def bn_extractor_all(model):

    d1_weights = []
    d1_bias = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bn_layer_list = [
        "features.7.0.bn1.weight",
        "features.7.0.bn1.bias",
        "features.7.0.bn2.weight",
        "features.7.0.bn2.bias",
        "features.7.1.bn1.weight",
        "features.7.1.bn1.bias",
        "features.7.1.bn2.weight",
        "features.7.1.bn2.bias",
    ]

    size_dict = {}
    for i in bn_layer_list:

        if "weight" in i:

            d1_weights.append(
                model.state_dict()[i].detach().clone().to(device)
            )
            size_dict[i] = model.state_dict()[i].shape[-1]
        elif "bias" in i:
            d1_bias.append(model.state_dict()[i].detach().clone().to(device))
            size_dict[i] = model.state_dict()[i].shape[-1]

    all_weight_tensors = d1_weights

    all_bias_tensors = d1_bias
    target_size = 512

    weights = torch.stack(all_weight_tensors)
    bias = torch.stack(all_bias_tensors)

    weights = weights.to(device)
    bias = bias.to(device)

    return weights, bias


def bn_extractor_gen2(model):

    d1_weights = []
    d1_bias = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bn_layer_list = [
        "features.1.weight",
        "features.1.bias",
        "features.4.0.bn1.weight",
        "features.4.0.bn1.bias",
        "features.4.0.bn2.weight",
        "features.4.0.bn2.bias",
        "features.4.1.bn1.weight",
        "features.4.1.bn1.bias",
        "features.4.1.bn2.weight",
        "features.4.1.bn2.bias",
        "features.5.0.bn1.weight",
        "features.5.0.bn1.bias",
        "features.5.0.bn2.weight",
        "features.5.0.bn2.bias",
        "features.5.0.downsample.1.weight",
        "features.5.0.downsample.1.bias",
        "features.5.1.bn1.weight",
        "features.5.1.bn1.bias",
        "features.5.1.bn2.weight",
        "features.5.1.bn2.bias",
        "features.6.0.bn1.weight",
        "features.6.0.bn1.bias",
        "features.6.0.bn2.weight",
        "features.6.0.bn2.bias",
        "features.6.0.downsample.1.weight",
        "features.6.0.downsample.1.bias",
        "features.6.1.bn1.weight",
        "features.6.1.bn1.bias",
        "features.6.1.bn2.weight",
        "features.6.1.bn2.bias",
    ]

    size_dict = {}
    for i in bn_layer_list:

        if not "downsample" in i:
            if "weight" in i:

                d1_weights.append(
                    model.state_dict()[i].detach().clone().to(device)
                )
                size_dict[i] = model.state_dict()[i].shape[-1]
            elif "bias" in i:
                d1_bias.append(
                    model.state_dict()[i].detach().clone().to(device)
                )
                size_dict[i] = model.state_dict()[i].shape[-1]

    all_weight_tensors = d1_weights

    all_bias_tensors = d1_bias
    target_size = 512

    weights = padding_adder(all_weight_tensors)
    bias = padding_adder(all_bias_tensors)

    return weights, bias


def padding_adder(weights):
    all_weight_gradients = weights
    target_size = 512

    for i, tensor in enumerate(all_weight_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_weight_gradients[i] = torch.nn.functional.pad(
                tensor, (0, pad_size), "constant", 0
            ).to(device)

    weights = torch.stack(all_weight_gradients, 0).to(device)

    return weights


def weights_generator(model, new_gamma, new_beta):

    added_weights = []
    added_bias = []

    added_weights.append(
        new_gamma[0][:512]
        + model.features._modules["7"]
        ._modules["0"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[0][:512]
        + model.features._modules["7"]
        ._modules["0"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[1][:512]
        + model.features._modules["7"]
        ._modules["0"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[1][:512]
        + model.features._modules["7"]
        ._modules["0"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[2][:512]
        + model.features._modules["7"]
        ._modules["1"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[2][:512]
        + model.features._modules["7"]
        ._modules["1"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[3][:512]
        + model.features._modules["7"]
        ._modules["1"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[3][:512]
        + model.features._modules["7"]
        ._modules["1"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights = torch.stack(added_weights)
    added_bias = torch.stack(added_bias)
    added_weights = added_weights.to(device)
    added_bias = added_bias.to(device)

    with torch.no_grad():

        model.features._modules["7"]._modules["0"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[0][:512])
        model.features._modules["7"]._modules["0"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[0][:512])
        model.features._modules["7"]._modules["0"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[1][:512])
        model.features._modules["7"]._modules["0"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[1][:512])

        model.features._modules["7"]._modules["1"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[2][:512])
        model.features._modules["7"]._modules["1"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[2][:512])
        model.features._modules["7"]._modules["1"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[3][:512])
        model.features._modules["7"]._modules["1"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[3][:512])

    return model, added_weights, added_bias


def weights_generator_gen2(model, new_gamma, new_beta):

    added_weights = []
    added_bias = []

    added_weights.append(
        new_gamma[0][:64] + model.features._modules["1"].weight.detach()
    )
    added_bias.append(
        new_beta[0][:64] + model.features._modules["1"].bias.detach()
    )

    added_weights.append(
        new_gamma[1][:64]
        + model.features._modules["4"]
        ._modules["0"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[1][:64]
        + model.features._modules["4"]
        ._modules["0"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[2][:64]
        + model.features._modules["4"]
        ._modules["0"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[2][:64]
        + model.features._modules["4"]
        ._modules["0"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[3][:64]
        + model.features._modules["4"]
        ._modules["1"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[3][:64]
        + model.features._modules["4"]
        ._modules["1"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[4][:64]
        + model.features._modules["4"]
        ._modules["1"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[4][:64]
        + model.features._modules["4"]
        ._modules["1"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[5][:128]
        + model.features._modules["5"]
        ._modules["0"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[5][:128]
        + model.features._modules["5"]
        ._modules["0"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[6][:128]
        + model.features._modules["5"]
        ._modules["0"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[6][:128]
        + model.features._modules["5"]
        ._modules["0"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[7][:128]
        + model.features._modules["5"]
        ._modules["1"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[7][:128]
        + model.features._modules["5"]
        ._modules["1"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[8][:128]
        + model.features._modules["5"]
        ._modules["1"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[8][:128]
        + model.features._modules["5"]
        ._modules["1"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[9][:256]
        + model.features._modules["6"]
        ._modules["0"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[9][:256]
        + model.features._modules["6"]
        ._modules["0"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[10][:256]
        + model.features._modules["6"]
        ._modules["0"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[10][:256]
        + model.features._modules["6"]
        ._modules["0"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights.append(
        new_gamma[11][:256]
        + model.features._modules["6"]
        ._modules["1"]
        ._modules["bn1"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[11][:256]
        + model.features._modules["6"]
        ._modules["1"]
        ._modules["bn1"]
        .bias.detach()
    )
    added_weights.append(
        new_gamma[12][:256]
        + model.features._modules["6"]
        ._modules["1"]
        ._modules["bn2"]
        .weight.detach()
    )
    added_bias.append(
        new_beta[12][:256]
        + model.features._modules["6"]
        ._modules["1"]
        ._modules["bn2"]
        .bias.detach()
    )

    added_weights = padding_adder(added_weights)
    added_bias = padding_adder(added_bias)

    with torch.no_grad():

        model.features._modules["1"].weight += nn.Parameter(new_gamma[0][:64])
        model.features._modules["1"].bias += nn.Parameter(new_beta[0][:64])

        model.features._modules["4"]._modules["0"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[1][:64])
        model.features._modules["4"]._modules["0"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[1][:64])
        model.features._modules["4"]._modules["0"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[2][:64])
        model.features._modules["4"]._modules["0"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[2][:64])

        model.features._modules["4"]._modules["1"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[3][:64])
        model.features._modules["4"]._modules["1"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[3][:64])
        model.features._modules["4"]._modules["1"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[4][:64])
        model.features._modules["4"]._modules["1"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[4][:64])

        model.features._modules["5"]._modules["0"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[5][:128])
        model.features._modules["5"]._modules["0"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[5][:128])
        model.features._modules["5"]._modules["0"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[6][:128])
        model.features._modules["5"]._modules["0"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[6][:128])

        model.features._modules["5"]._modules["1"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[7][:128])
        model.features._modules["5"]._modules["1"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[7][:128])
        model.features._modules["5"]._modules["1"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[8][:128])
        model.features._modules["5"]._modules["1"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[8][:128])

        model.features._modules["6"]._modules["0"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[9][:256])
        model.features._modules["6"]._modules["0"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[9][:256])
        model.features._modules["6"]._modules["0"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[10][:256])
        model.features._modules["6"]._modules["0"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[10][:256])

        model.features._modules["6"]._modules["1"]._modules[
            "bn1"
        ].weight += nn.Parameter(new_gamma[11][:256])
        model.features._modules["6"]._modules["1"]._modules[
            "bn1"
        ].bias += nn.Parameter(new_beta[11][:256])
        model.features._modules["6"]._modules["1"]._modules[
            "bn2"
        ].weight += nn.Parameter(new_gamma[12][:256])
        model.features._modules["6"]._modules["1"]._modules[
            "bn2"
        ].bias += nn.Parameter(new_beta[12][:256])

    return model, added_weights, added_bias


def gradients_extract(d1):
    weights = []

    d1 = {k: v.detach() for k, v in d1.items()}

    all_weights = []
    all_bias = []

    for k, v in d1.items():

        if "7" in k:
            if "downsample" not in k:
                if "weight" in k:
                    all_weights.append(v)

                if "bias" in k:
                    all_bias.append(v)

    all_weight_gradients = torch.stack(all_weights).to(device)

    all_bias_gradients = torch.stack(all_bias).to(device)

    return all_weight_gradients, all_bias_gradients


def gradients_extract_gen2(d1):
    weights = []

    d1 = {k: v.detach() for k, v in d1.items()}

    all_weights = []
    all_bias = []

    for k, v in d1.items():

        if not "7" in k:
            if "downsample" not in k:
                if "weight" in k:
                    all_weights.append(v)

                if "bias" in k:
                    all_bias.append(v)

    all_weight_gradients = all_weights
    target_size = 512

    for i, tensor in enumerate(all_weight_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_weight_gradients[i] = torch.nn.functional.pad(
                tensor, (0, pad_size), "constant", 0
            ).to(device)

    all_bias_gradients = all_bias

    for i, tensor in enumerate(all_bias_gradients):
        current_size = tensor.shape[-1]
        pad_size = target_size - current_size

        if pad_size > 0:
            all_bias_gradients[i] = torch.nn.functional.pad(
                tensor, (0, pad_size), "constant", 0
            ).to(device)

    weights = torch.stack(all_weight_gradients, 0).to(device)
    bias = torch.stack(all_bias_gradients, 0).to(device)

    return weights, bias


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_tent_model(model):
    """Configure model for use with tent."""

    model.train()

    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)

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
                if np in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def train_meta_target(epoch, domain_id, scores):

    net.train()
    tr_net.train()
    tr_net_gen2.train()

    del scores

    tr_net.train()

    t0 = time.time()

    loss_model_gen = 0

    log_string("\n ")
    log_string("Domain ID %d" % domain_id)

    rt_context.reset("train", domain_id, transform=transform_train)

    context_loader = torch.utils.data.DataLoader(
        rt_context,
        batch_size=(num_domain - 1) * NUM_CLASS * ctx_num,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    softmax_loss = 0

    net_copy = copy.deepcopy(net)
    net_copy.to(device)
    net_copy.train()

    net_copy = configure_tent_model(net_copy)

    tent_params, tent_names = collect_tent_params(net_copy)

    optimizer_tent = torch.optim.AdamW(
        tent_params, lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    net_original = copy.deepcopy(net)
    net_original.to(device)
    net_original.train()
    net_original = configure_tent_model(net_original)
    tent_params_original, tent_names_original = collect_tent_params(
        net_original
    )

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, img_name2) in enumerate(context_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_tent.zero_grad()

        outputs, _ = net_copy(inputs)

        loss_sfm = softmax_entropy(outputs).mean(0)
        softmax_loss += loss_sfm.item()
        loss_sfm.backward()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        scores = {}
        bn_layer_list = []
        for name, m in net_copy.named_modules():

            if isinstance(m, torch.nn.BatchNorm2d):

                scores[name + ".weight"] = torch.clone(
                    m.weight.grad.clone()
                ).detach()
                scores[name + ".bias"] = torch.clone(
                    m.bias.grad.clone()
                ).detach()
                bn_layer_list.append(name)

        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])

        grad_flow = torch.norm(all_scores)
        optimizer_tent.step()

    log_string("Softmax loss: %f" % (softmax_loss / (batch_idx + 1)))
    log_string(
        "Accuracy through entropy minimization: %f (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )

    wandb.log(
        {
            "meta_target_acc/softmax_loss": softmax_loss / (batch_idx + 1),
            "epoch": epoch,
        }
    )
    wandb.log({"meta_target_acc/acc": 100.0 * correct / total, "epoch": epoch})

    grad_update_wandb = {}

    loss_softmax_ent = 0
    total = 0
    correct = 0
    train_loss = 0
    loss_numerical_tot = 0
    total_source_model = 0
    correct_source_model = 0

    for batch_idx, (inputs, targets, img_name2) in enumerate(context_loader):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_tr_net.zero_grad()

        _, outputs_central = net(inputs)

        weight, bias = bn_extractor_all(net)

        weights_bias = torch.cat((weight, bias), dim=0).to(device)

        all_weight, all_bias = gradients_extract(scores)

        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)

        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()

        outputs_central_mean = outputs_central.mean(0, keepdim=True)
        del outputs_central

        outputs_central_mean = outputs_central_mean.repeat(8, 1)

        outputs_tr_net = tr_net(
            outputs_central_mean, all_gradients, weights_bias
        )

        new_gamma = outputs_tr_net[:4]
        new_beta = outputs_tr_net[4:]
        new_gb = torch.cat((new_gamma, new_beta), dim=0).to(device)

        net_original, added_weights, added_bias = weights_generator(
            net_original, new_gamma, new_beta
        )
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(
            device
        )

        tent_weights, tent_bias = bn_extractor_all(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(
            device
        )

        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent += loss_softmax.item()

        train_loss += loss_tr_net.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        outputs_source_model, _ = net(inputs)
        _, predicted_source_model = outputs_source_model.max(1)
        total_source_model += targets.size(0)
        correct_source_model += predicted_source_model.eq(targets).sum().item()

        loss_numerical = criterion_numerical(
            added_weights_bias, tent_weights_bias
        )
        loss_numerical_tot += loss_numerical.item()

        total_loss = loss_tr_net + loss_numerical

        total_loss.backward()
        optimizer_tr_net.step()

    log_string(
        "Accuracy of Model Generated: %.3f%% (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )
    log_string(
        "Accuracy of Source Model: %.3f%% (%d/%d)"
        % (
            100.0 * correct_source_model / total_source_model,
            correct_source_model,
            total_source_model,
        )
    )
    log_string("Loss Numerical: %.3f" % (loss_numerical_tot / (batch_idx + 1)))
    log_string("Epoch: %d, Loss: %.3f" % (epoch, train_loss / (batch_idx + 1)))

    wandb.log({"train/loss": train_loss / (batch_idx + 1)})
    wandb.log({"epoch": epoch})
    wandb.log({"train/acc": 100.0 * correct / total})
    wandb.log({"train/lr": optimizer.param_groups[0]["lr"]})
    wandb.log({"meta_target/tr_net_loss": loss_model_gen / (batch_idx + 1)})

    wandb.log(
        {
            "meta_target_acc/source_acc ": 100.0
            * correct_source_model
            / total_source_model
        }
    )
    wandb.log({"meta_target_acc/model_gen_acc": 100.0 * correct / total})
    wandb.log({"meta_target_acc/source_loss": train_loss / (batch_idx + 1)})
    wandb.log(
        {
            "meta_target_acc/model_gen_loss": loss_numerical_tot
            / (batch_idx + 1)
        }
    )
    wandb.log({"epoch": epoch})

    wandb.log({"meta_target/loss": train_loss / (batch_idx + 1)})
    wandb.log({"meta_target/acc": 100.0 * correct / total})
    wandb.log({"meta_target/lr": optimizer.param_groups[0]["lr"]})

    loss_softmax_ent = 0
    total = 0
    correct = 0
    train_loss = 0
    loss_numerical_tot = 0
    total_source_model = 0
    correct_source_model = 0

    tr_net_gen2.train()

    for batch_idx, (inputs, targets, img_name2) in enumerate(context_loader):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer_tr_net_gen2.zero_grad()

        _, outputs_central = net(inputs)

        weight, bias = bn_extractor_gen2(net_original)

        weights_bias = torch.cat((weight, bias), dim=0).to(device)

        all_weight, all_bias = gradients_extract_gen2(scores)

        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)

        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()

        outputs_central_mean = outputs_central.mean(0, keepdim=True)
        del outputs_central

        outputs_central_mean = outputs_central_mean.repeat(26, 1)

        outputs_tr_net_gen2 = tr_net_gen2(
            outputs_central_mean, all_gradients, weights_bias
        )

        new_gamma = outputs_tr_net_gen2[:13]
        new_beta = outputs_tr_net_gen2[13:]
        new_gb = torch.cat((new_gamma, new_beta), dim=0).to(device)

        net_original, added_weights, added_bias = weights_generator_gen2(
            net_original, new_gamma, new_beta
        )
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(
            device
        )

        tent_weights, tent_bias = bn_extractor_gen2(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(
            device
        )

        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent += loss_softmax.item()

        train_loss += loss_tr_net.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        outputs_source_model, _ = net(inputs)
        _, predicted_source_model = outputs_source_model.max(1)
        total_source_model += targets.size(0)
        correct_source_model += predicted_source_model.eq(targets).sum().item()

        gamma_beta = torch.cat((new_gamma, new_beta), dim=0)

        loss_numerical = criterion_numerical(
            added_weights_bias, tent_weights_bias
        )
        loss_numerical_tot += loss_numerical.item()

        total_loss = loss_tr_net + loss_numerical

        total_loss.backward()

        optimizer_tr_net_gen2.step()

    log_string(
        "Accuracy of Model Generated: %.3f%% (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )
    log_string(
        "Accuracy of Source Model: %.3f%% (%d/%d)"
        % (
            100.0 * correct_source_model / total_source_model,
            correct_source_model,
            total_source_model,
        )
    )
    log_string("Loss Numerical: %.3f" % (loss_numerical_tot / (batch_idx + 1)))
    log_string("Epoch: %d, Loss: %.3f" % (epoch, train_loss / (batch_idx + 1)))
    wandb.log({"meta_target_acc/model_gen2_acc": 100.0 * correct / total})
    wandb.log(
        {
            "meta_target_acc/model_gen2_loss": loss_numerical_tot
            / (batch_idx + 1)
        }
    )
    wandb.log({"epoch": epoch})

    del net_copy, net_original
    tr_net.eval()
    tr_net_gen2.eval()

    print("time elapsed: %f" % (time.time() - t0))


NUM_BATCHES_TO_LOG = 2


def test(epoch):

    global best_acc
    global best_valid_acc

    log_counter = 0

    net.eval()
    tr_net.eval()
    tr_net_gen2.eval()

    log_string("-----------------Test----------------- \n")

    all_dataset.reset("test", 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=cpu_workers,
        worker_init_fn=worker_init_fn,
    )

    tr_net_copy = copy.deepcopy(tr_net)
    tr_net_copy.to(device)
    tr_net_copy.train()

    net_original = copy.deepcopy(net)
    net_original.to(device)
    net_original.train()

    net_copy = copy.deepcopy(net)
    net_copy.to(device)
    net_copy.train()

    net_copy = configure_tent_model(net_copy)
    net_original = configure_tent_model(net_original)

    tent_params, tent_names = collect_tent_params(net_copy)

    optimizer_tent_copy = torch.optim.AdamW(
        tent_params, lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    total = 0
    correct = 0
    softmax_loss = 0

    for batch_idx, (inputs, targets, img_name2) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_tent_copy.zero_grad()

        outputs, _ = net_copy(inputs)

        loss_sfm = softmax_entropy(outputs).mean(0)
        softmax_loss += loss_sfm.item()
        loss_sfm.backward()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        scores = {}
        bn_layer_list = []
        for name, m in net_copy.named_modules():

            if isinstance(m, torch.nn.BatchNorm2d):

                scores[name + ".weight"] = torch.clone(
                    m.weight.grad.clone()
                ).detach()
                scores[name + ".bias"] = torch.clone(
                    m.bias.grad.clone()
                ).detach()
                bn_layer_list.append(name)

        all_scores = torch.cat([torch.flatten(v) for v in scores.values()])

        grad_flow = torch.norm(all_scores)
        optimizer_tent_copy.step()

    log_string("\tSoftmax loss: %f" % (softmax_loss / (batch_idx + 1)))
    log_string(
        "\tAccuracy through entropy: %.3f%% (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )
    grad_update_wandb = {}

    log_string("\n ")

    loss_softmax_ent = 0
    total = 0
    correct = 0
    test_loss = 0
    model_gen_numerical = 0

    total_source_model = 0
    correct_source_model = 0

    for batch_idx, (inputs, targets, img_name2) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)

        _, outputs_central = net_original(inputs)

        weight, bias = bn_extractor_all(net_original)

        weights_bias = torch.cat((weight, bias), dim=0).to(device)

        all_weight, all_bias = gradients_extract(scores)

        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)

        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()

        outputs_central_mean = outputs_central.mean(0, keepdim=True)
        del outputs_central

        outputs_central_mean = outputs_central_mean.repeat(8, 1)

        outputs_tr_net = tr_net_copy(
            outputs_central_mean, all_gradients, weights_bias
        )

        new_gamma = outputs_tr_net[:4]
        new_beta = outputs_tr_net[4:]

        net_original, added_weights, added_bias = weights_generator(
            net_original, new_gamma, new_beta
        )
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(
            device
        )

        tent_weights, tent_bias = bn_extractor_all(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(
            device
        )

        outputs, _ = net_original(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent += loss_softmax.item()

        gamma_beta = torch.cat((new_gamma, new_beta), dim=0).to(device)

        loss_numerical = criterion_numerical(
            added_weights_bias, tent_weights_bias
        )
        model_gen_numerical += loss_numerical.item()

        total_loss = loss_numerical
        t_ls = criterion(outputs, targets)
        test_loss += t_ls.item()

    log_string("\tSoftmax loss: %f" % (loss_softmax_ent / (batch_idx + 1)))

    log_string(
        "\tAccuracy of model gen : %.3f%% (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )
    log_string("\tTest loss: %f" % (test_loss / (batch_idx + 1)))
    log_string(
        "\tModel gen loss numerical : %f"
        % (model_gen_numerical / (batch_idx + 1))
    )

    wandb.log({"test/model_gen_acc": 100.0 * correct / total, "epoch": epoch})
    wandb.log(
        {
            "test/model_gen_loss": model_gen_numerical / (batch_idx + 1),
            "epoch": epoch,
        }
    )
    acc_tr_net1 = 100.0 * correct / total

    del tr_net_copy

    del (
        inputs,
        targets,
        outputs,
        outputs_central_mean,
        outputs_tr_net,
        predicted,
        gamma_beta,
        new_gamma,
        new_beta,
        all_gradients,
        weights_bias,
        all_weight,
        all_bias,
        weight,
        bias,
        tent_weights,
        tent_bias,
        tent_weights_bias,
        added_weights,
        added_bias,
        added_weights_bias,
    )

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

    for batch_idx, (inputs, targets, img_name2) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)

        _, outputs_central = net_original(inputs)

        weight, bias = bn_extractor_gen2(net_original)

        weights_bias = torch.cat((weight, bias), dim=0).to(device)

        all_weight, all_bias = gradients_extract_gen2(scores)

        all_gradients = torch.cat((all_weight, all_bias), dim=0).to(device)

        all_gradients = all_gradients.detach()
        outputs_central = outputs_central.detach()

        outputs_central_mean = outputs_central.mean(0, keepdim=True)
        del outputs_central

        outputs_central_mean = outputs_central_mean.repeat(26, 1)

        outputs_tr_net = tr_net_gen2_copy(
            outputs_central_mean, all_gradients, weights_bias
        )

        new_gamma = outputs_tr_net[:13]
        new_beta = outputs_tr_net[13:]

        net_original, added_weights, added_bias = weights_generator_gen2(
            net_original, new_gamma, new_beta
        )
        added_weights_bias = torch.cat((added_weights, added_bias), dim=0).to(
            device
        )

        tent_weights, tent_bias = bn_extractor_gen2(net_copy)
        tent_weights_bias = torch.cat((tent_weights, tent_bias), dim=0).to(
            device
        )

        outputs, _ = net_original(inputs)

        loss_tr_net = criterion(outputs, targets)
        loss_softmax = softmax_entropy(outputs).mean(0)
        loss_softmax_ent += loss_softmax.item()

        test_loss += loss_tr_net.item()

        _, predicted = outputs.max(1)
        del outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        gamma_beta = torch.cat((new_gamma, new_beta), dim=0)

        loss_numerical = criterion_numerical(
            added_weights_bias, tent_weights_bias
        )
        loss_numerical_tot += loss_numerical.item()

    log_string("\tSoftmax loss: %f" % (loss_softmax_ent / (batch_idx + 1)))

    log_string(
        "\tAccuracy of model gen : %.3f%% (%d/%d)"
        % (100.0 * correct / total, correct, total)
    )
    log_string("\tTest loss: %f" % (test_loss / (batch_idx + 1)))
    log_string(
        "\tModel gen loss numerical : %f"
        % (loss_numerical_tot / (batch_idx + 1))
    )

    wandb.define_metric("test/model_gen2_acc", step_metric="epoch")
    wandb.define_metric("test/model_gen2_loss", step_metric="epoch")
    wandb.log(
        {
            "test/model_gen2_acc": 100.0 * correct / total,
            "test/model_gen2_loss": loss_numerical_tot / (batch_idx + 1),
        }
    )
    wandb.log({"epoch": epoch})

    acc_tr_net2 = 100.0 * correct / total

    acc = max(acc_tr_net1, acc_tr_net2)
    if acc > best_valid_acc:
        best_valid_acc = acc

        print("Saving best model... %f" % acc)
        state = {
            "net": net.state_dict(),
            "tr_net": tr_net.state_dict(),
            "tr_net_gen2": tr_net_gen2.state_dict(),
            "model_gen_acc": acc,
            "epoch": epoch,
        }

        checkpoint_dir = os.path.join(MODEL_DIR, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, "best_model.pth"))
        wandb.summary["model_path"] = os.path.join(
            checkpoint_dir, "best_model.pth"
        )
        wandb.summary["model_path_epoch"] = epoch


def normal_test(epoch):
    global best_acc

    log_counter = 0

    net.eval()
    tr_net.eval()

    all_dataset.reset("test", 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=cpu_workers,
        worker_init_fn=worker_init_fn,
    )

    test_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    batch_count = 0
    """with torch.no_grad():

        for batch_idx, (inputs, targets,  img_name1 ) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _= net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()"""

    with torch.no_grad():

        for batch_idx, (inputs, targets, img_name1) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        log_string(
            "\t Test Loss %f, Acc: %f"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total)
        )

        wandb.log({"normal_test/acc": 100.0 * correct / total})
        wandb.log({"normal_test/loss": test_loss / batch_idx + 1})
        wandb.log({"epoch": epoch})

    """acc = 100.*correct/total
    if acc > best_valid_acc:
        print('Saving best model... %f' % acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pth'))"""


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

    with torch.no_grad():
        for i in range(4):
            all_dataset.reset("val", i, transform=transform_test)
            valloader = torch.utils.data.DataLoader(
                all_dataset,
                batch_size=test_batch,
                shuffle=False,
                num_workers=4,
            )

            num_preds = 1

            for batch_idx, (inputs, targets, _) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    log_string(
        "VAL Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (
            test_loss / (batch_idx + 1),
            100.0 * correct / total,
            correct,
            total,
        )
    )

    wandb.log({"val/acc": 100.0 * correct / total})
    wandb.log({"val/loss": test_loss / (batch_idx + 1)})
    wandb.log({"epoch": epoch})

    return 0


decay_ite = [0.6 * max_ite]

if args.autodecay:
    for epoch in range(300):
        train(epoch)
        f = test(epoch)
        if f == 0:
            converge_count = 0
        else:
            converge_count += 1

        if converge_count == 20:
            optimizer.param_groups[0]["lr"] = (
                optimizer.param_groups[0]["lr"] * 0.2
            )
            log_string(
                "In epoch %d the LR is decay to %f"
                % (epoch, optimizer.param_groups[0]["lr"])
            )
            converge_count = 0

        if optimizer.param_groups[0]["lr"] < 2e-6:
            exit()

else:
    if not iteration_training:
        for epoch in range(start_epoch, 10000000000000):
            if epoch in decay_inter:
                optimizer.param_groups[0]["lr"] = (
                    optimizer.param_groups[0]["lr"] * 0.1
                )
                log_string(
                    "In epoch %d the LR is decay to %f"
                    % (epoch, optimizer.param_groups[0]["lr"])
                )
            train(epoch)

            train_meta_target(epoch)
            if epoch % 5 == 0:
                _ = validation(epoch)
                _ = test(epoch)
                normal_test(epoch)
    else:
        for epoch in range(10000000000):
            if epoch in decay_ite:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["lr"] = (
                        optimizer.param_groups[i]["lr"] * 0.1
                    )
                log_string(
                    "In iteration %d the LR is decay to %f"
                    % (epoch, optimizer.param_groups[0]["lr"])
                )
            domain_id, scores = train(epoch)
            train_meta_target(epoch, domain_id, scores)

            scheduler_tr_net.step()
            scheduler_tr_net_gen2.step()

            if epoch > 200 and epoch % 30 == 0:

                _ = test(epoch)
                normal_test(epoch)

            if epoch % 250 == 0:
                _ = test(epoch)
                normal_test(epoch)
