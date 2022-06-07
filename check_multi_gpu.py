# import os
# import sys
# import tempfile
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms

# from torch.nn.parallel import DistributedDataParallel as DDP

# transform = transforms.Compose([transforms.ToTensor()])


# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# # train_batch_sampler = torch.utils.data.BatchSampler(
# #         train_sampler, batch_size, drop_last=True)

# # trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, batch_size=batch_size,
# #                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
#     global train_sampler
#     train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    
#     global train_batch_sampler
#     train_batch_sampler = torch.utils.data.BatchSampler(
#             train_sampler, batch_size, drop_last=True)

#     global trainloader
#     trainloader = torch.utils.data.DataLoader(trainset, sampler=train_batch_sampler, batch_size=batch_size,
#                                              num_workers=2)


# def cleanup():
#     dist.destroy_process_group()


# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(rank, world_size):
#     print(f"Running basic DDP example on rank {rank}.")
#     setup(rank, world_size)

#     # create model and move it to GPU with id rank
#     model = Net().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])

#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    


#     for epoch in range(2):
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data[0].to(rank), data[1].to(rank)
#             optimizer.zero_grad()
#             outputs = ddp_model(inputs)
#             # labels = torch.randn(20, 5).to(rank)
#             loss_fn(outputs, labels).backward()
#             optimizer.step()
#             running_loss += loss_fn(outputs, labels)
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#                 running_loss = 0.0

#     # optimizer.zero_grad()
#     # outputs = ddp_model(torch.randn(20, 10))
#     # labels = torch.randn(20, 5).to(rank)
#     # loss_fn(outputs, labels).backward()
#     # optimizer.step()
#     if(os.path.exists("CIFAR_models")==False):
#         os.mkdir("CIFAR_models")
#     torch.save(ddp_model, "./CIFAR_models/wo_batchnorm.model")
#     print('Finished Training')
#     cleanup()


# def run_demo(demo_fn, world_size):
#     mp.spawn(demo_fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)

# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
#     world_size = n_gpus
#     run_demo(demo_basic, world_size)

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim

# # Parameters and DataLoaders
# input_size = 5
# output_size = 2

# batch_size = 30
# data_size = 100


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class RandomDataset(Dataset):

#     def __init__(self, size, length):
#         self.len = length
#         self.data = torch.randn(length, size)

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return self.len

# rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
#                          batch_size=batch_size, shuffle=True)


# class Model(nn.Module):
#     # Our model

#     def __init__(self, input_size, output_size):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input):
#         return self.sigmoid(self.fc(input))


# model = Model(input_size, output_size)
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)

# model.to(device)

# optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
# cls_criterion = nn.CrossEntropyLoss()

# for data in rand_loader:
#     targets = torch.empty(data.size(0)).random_(2).view(-1, 1).to(device)
#     input = data.to(device)
#     output = model(input)
#     print("Outside: input size", input.size(),
#           "output_size", output.size())


# Source: https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
# Source: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='gloo',  world_size=args.world_size, rank=rank) #init_method='env://',
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()#.cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print(i)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            # print("output pass")
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()