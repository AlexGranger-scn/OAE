import os
import argparse
import torch
torch.cuda.current_device()
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#from tensorboardx import SummaryWriter
import pdb
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
from model import *
from dataset import CustomDataset
from torch.optim.lr_scheduler import StepLR


def train(epoch, dataloader, net, optimizer , mse_loss,device):
    accum_loss = 0
    net.train()
    for id, data in enumerate(dataloader):
        data= data.to(device)

        net.zero_grad()
        hidden_layer, output = net(data)
        loss = mse_loss(data,output)
        
        loss.backward()
        optimizer.step()
        #accum_loss += loss.data[0]
        accum_loss += loss.item()

        print(f'[{epoch}][{id}/{len(dataloader)}] MSE_loss: {loss.item():.4f}')
        print("%d epoch lr: %f" % (epoch, optimizer.param_groups[0]['lr']))
        
    return accum_loss / len(dataloader)


def test(epoch, dataloader, net , mse_loss,device):
    accum_loss = 0
    net.eval()
    for id, data in enumerate(dataloader):
        data = data.to(device)

        hidden_layer, output = net(data)
        loss = mse_loss(data,output)
        
        accum_loss += loss.item()
    
    accum_loss /= len(dataloader)
    print(f'[{epoch}] val loss: {accum_loss:.4f}')
    return accum_loss

    
def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.device(i_gpu).__enter__()
    cudnn.benchmark = True


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description='train OrderedAutoEncoder binary')
    parser.add_argument('--weights', default='', help='path to weight (to continue training)')
    parser.add_argument('--outf', default=None, help='path to store checkpoint')
    parser.add_argument('--checkpoint', type=int, default=None, help='checkpointing after batches')
    parser.add_argument('--data_npy_path',type=str,default=None, help='path of x-vector')
    parser.add_argument('--train_data_split',type=str,default=None, help='randomly select 80% of data for training')
    parser.add_argument('--val_data_split',type=str,default=None, help='randomly select 20% of data for validation')

    parser.add_argument('--batchSize', type=int, default=None, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=None, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=None, help='length of hashing binary')

    parser.add_argument('--niter', type=int, default=None, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--log_path',type=str,default=None, 'record of training loss')
    args = parser.parse_args()
    print(args)

    choose_gpu(args.ngpu)
    feed_random_seed()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_emb_dict=np.load(args.data_npy_path,allow_pickle=True).item()
    train_loader = DataLoader(CustomDataset(args.train_data_split,data_emb_dict),batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(CustomDataset(args.val_data_split,data_emb_dict),batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)


    # setup net
    net = OrderedAutoEncoder_b(args.binary_bits)
    resume_epoch = 0
    print(net)
    if args.weights: 
        print(f'loading weight form {args.weights}')
        resume_epoch = int(os.path.basename(args.weights)[:-4])
        net.load_state_dict(torch.load(args.weights, map_location=lambda storage, location: storage))

    net.cuda()

    val_loss=[]
    lr=[]

    # setup optimizer
    MSE_loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.004)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    
    log_path = args.log_path + "/loss.log"

    for epoch in range(resume_epoch, args.niter):

        train(epoch, train_loader, net, optimizer , MSE_loss, device)
        test_loss = test(epoch, test_loader, net , MSE_loss, device)
        val_loss.append(test_loss)

        lr.append(optimizer.param_groups[0]['lr'])

        scheduler.step()

        with open(log_path,'a') as f:
            f.write("epoch:"+str(epoch)+'\n')
            f.write(str(test_loss)+'\n')
        
        if epoch % args.checkpoint == 0 and epoch>= 10:
            torch.save(net.state_dict(), os.path.join(args.outf, f'{epoch:03d}.pth'))

    with open(log_path,'a') as f:
        f.write("val_loss:"+'\n')
        f.write(str(val_loss)+'\n')
        f.write("lr:"+'\n')
        f.write(str(lr))
        
if __name__ == '__main__':
    main()
