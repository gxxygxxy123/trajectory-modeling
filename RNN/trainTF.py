import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import sys

sns.set()
import individual_TF
from utils import predict, param, error_function
from dataloader import RNNDataSet, PhysicsDataSet

DIRNAME = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.dirname(DIRNAME)
sys.path.append(f"{ROOTDIR}/lib")

from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt

DXYT = True

def train(model, iterator, optimizer, criterion, mean, std, device):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):

        optimizer.optimizer.zero_grad()

        inp=(batch['src'][:,:].to(device)-mean.to(device))/std.to(device)
        target=(batch['trg'][:,:-1].to(device)-mean.to(device))/std.to(device)

        target_c = torch.zeros((target.shape[0],target.shape[1],1)).to(device)
        target = torch.cat((target,target_c),-1)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

        dec_inp = torch.cat((start_of_seq, target), 1)

        src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

        pred = model(inp, dec_inp, src_att, trg_att)

        loss = criterion(pred[:, :,0:2]
        .contiguous().view(-1, 2), ((batch['trg'][:, :, :]
        .to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
                        ).mean() + torch.mean(torch.abs(pred[:,:,2])) # this is neccessary !!! but why?

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    #     if batch_idx == 0:
    #         fig,ax = plt.subplots()
    #         if DXYT:
    #             preds=((pred[:,:,[0,1]].cpu()*std+mean).cumsum(1) + (inp.cpu()*std+mean).cumsum(1)[:,-1:,:]).detach().numpy()
    #             tmp1 = (inp.cpu()*std+mean).cumsum(1).detach().numpy()
    #             tmp2 = batch['trg'].cumsum(1).detach().numpy() + tmp1[:,-1:,:]
    #         else:
    #             preds=((pred[:,:,[0,1]].cpu()*std+mean) + inp[:,0:1,[0,1]].cpu()).detach().numpy()
    #             tmp1 = (inp.cpu()*std+mean).detach().numpy()
    #             tmp2 = batch['trg'].detach().numpy()
    #         #tmp3 = ((pred[:,:,[0,1]].cpu()*std+mean).cumsum(1).detach().numpy() + tmp1[:,-1:,:])
    #         for i in range(10):
    #             pad_t = np.expand_dims(np.arange(0, batch['trg'].shape[1]), axis=1)
    #             pr = np.concatenate((preds[i],pad_t),axis=1)
    #             if DXYT:
    #                 gt = np.concatenate(((batch['trg'][i].cumsum(0)+(inp[i].cpu()*std+mean).cumsum(0)[-1:,:]).detach().numpy(),pad_t),axis=1)
    #             else:
    #                 gt = np.concatenate(((batch['trg'][i]+(inp[i].cpu()*std+mean)[0:1,:]).detach().numpy(),pad_t),axis=1)
    #             p = ax.plot(tmp1[i,:,0], tmp1[i,:,1], marker='o', markersize=1)
    #             ax.plot(tmp2[i,:,0], tmp2[i,:,1], color=p[0].get_color(), linestyle='--')
    #             ax.plot(pr[:,0], pr[:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())
    #             train_loss_debug.append(error_function.space_time_err(gt,pr))
    #         fig.savefig(os.path.join(args.fig_pth, f'{epoch}_train.png'),dpi=200)
    #         ax.clear()
    #         plt.close(fig)
    # print(f"Train Debug Loss : {np.nanmean(train_loss_debug)}")
    return epoch_loss / len(iterator)

def inference(model, dataset, mean, std, device, epoch, N):

    model.eval()

    infer_loss = []

    fig,ax = plt.subplots()
    ax.set_title(f"Epoch {epoch}, input {N} points, {round(dataset.fps())} FPS")
    ax.set_xlabel(f"distance (m)")
    ax.set_ylabel(f"height (m)")

    with torch.no_grad():

        for idx, trajectory in dataset.whole_2d().items():
            if trajectory.shape[0] <= N:
                continue

            inp = trajectory[:N].copy()
            gt = trajectory[N:].copy()

            out = predict.predict2d_TF(inp, model, mean=mean, std=std, out_time=2.0, fps=dataset.fps(), touch_ground_stop=True, device=device, dxyt=DXYT)

            if args.draw_inference:
                p = ax.plot(inp[:,0], inp[:,1], marker='o', markersize=1)
                ax.plot(gt[:,0], gt[:,1], color=p[0].get_color(), linestyle='--')
                ax.plot(out[inp.shape[0]:,0], out[inp.shape[0]:,1], marker='o', markersize=1, alpha=0.3, color=p[0].get_color())
                ax.scatter(gt[::10,0],gt[::10,1], color='red',s=2)
                ax.scatter(out[inp.shape[0]::10,0],out[inp.shape[0]::10,1], color='blue',s=2)

            infer_loss.append(error_function.space_time_err(gt,out[inp.shape[0]:]))

        if epoch % 1 == 0:
            fig.savefig(os.path.join(args.fig_pth, f'{epoch}_{N}.png'),dpi=200)
            ax.clear()


    plt.close(fig)

    return sum(infer_loss) / len(infer_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Training Program")
    parser.add_argument("-t","--time", type=float, help="Input Sequence Time", default=0.1)
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", default=300)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=100000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=32)
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=1)
    parser.add_argument('--fig_pth', type=str, default='./figure/TF/')
    parser.add_argument('--wgt_pth', type=str, default='./weight/TF/')
    parser.add_argument('--noisexy', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.2151959552)
    parser.add_argument('--draw_inference', action="store_true", default=False)

    parser.add_argument('--emb_size',type=int,default=128) # 512
    parser.add_argument('--heads',type=int, default=4) # 8
    parser.add_argument('--layers',type=int,default=2) # 6
    parser.add_argument('--d_ff',type=int, default=512) # 2048
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=10)


    args = parser.parse_args()

    os.makedirs(args.fig_pth, exist_ok=True)
    os.makedirs(args.wgt_pth, exist_ok=True)


    print(f"Epoch: {args.epoch}\n"
          f"Physics data: {args.physics_data}\n"
          f"Batch Size: {args.batch_size}\n"
          f"emb_size:{args.emb_size},heads:{args.heads},layers:{args.layers},d_ff:{args.d_ff},factor:{args.factor},warmup:{args.warmup}\n"
          f"Save At Each {args.save_epoch} epoch\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=args.physics_data, model='TF', in_max_time=args.time, dxyt=DXYT, network_in_dim=2, 
                                   drop_mode=0, out_max_time=2.0, noise_xy=args.noisexy, alpha=args.alpha)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last=True)
    print(f"Training data length: {len(train_dataset)}")

    # Inference Dataset
    infer_dataset = RNNDataSet(dataset_path="../trajectories_dataset/", fps=120, smooth_2d=True)

    #print("Inference Last Experiment (Overfitting)")
    #infer_dataset = PhysicsDataSet(in_max_time=args.time, out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')

    model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                   d_model=args.emb_size, d_ff=args.d_ff, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)
    print(f'The model has {param.count_parameters(model):,} trainable parameters')
    #model.apply(param.init_weights)

    optimizer = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                        optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    criterion = F.pairwise_distance

    mean = train_dataset.mean()
    std = train_dataset.std()

    print(f"Train dataset mean:{mean}, std:{std}")
    hist_tr_loss = []

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,args.epoch+1):

        tr_loss = train(model, tr_dl, optimizer, criterion, mean=mean, std=std, device=device)
        hist_tr_loss.append(tr_loss)

        infer_loss = inference(model, infer_dataset, mean=mean, std=std, device=device, epoch=epoch, N=math.floor(infer_dataset.fps()*args.time)+1)

        print(f"Epoch: {epoch}/{args.epoch}. Train Loss: {tr_loss:.8f}. Infer Loss: {infer_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % args.save_epoch == 0:
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.optimizer.state_dict(),
                        'datas':args.physics_data,
                        'mean':mean,
                        'std':std,
                        'emb_size':args.emb_size,
                        'heads':args.heads,
                        'layers':args.layers,
                        'd_ff':args.d_ff,
                       },  os.path.join(args.wgt_pth, f'TF_p{args.physics_data}_e{epoch}'))
            # print(f"Save Weight At Epoch {epoch}")


    torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.optimizer.state_dict(),
                'datas':args.physics_data,
                'mean':mean,
                'std':std,
                'emb_size':args.emb_size,
                'heads':args.heads,
                'layers':args.layers,
                'd_ff':args.d_ff,
                },  os.path.join(args.wgt_pth, f'TF_final'))

    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist_tr_loss)+1) , hist_tr_loss)
    ax.set_title('Transformer')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')
    plt.show()
