import torch.nn as nn
import torch
import pandas as pd
import os
import sys
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from datetime import datetime

from blstm import Blstm
from threeDprojectTo2D import FitPlane, ProjectToPlane, ThreeDPlaneTo2DCoor, fit_3d, fit_2d, FitVerticalPlaneTo2D
from dataloader import RNNDataSet, PhysicsDataSet
from utils import predict, param, error_function
sns.set()

def train(model, iterator, optimizer, criterion, mean, std, device):
    
    model.train()
    
    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):
        src=(batch['src'].to(device)-mean.to(device))/std.to(device)
        trg=(batch['trg'].to(device)-mean.to(device))/std.to(device)

        src_lens = batch['src_lens']
        trg_lens = batch['trg_lens']

        optimizer.zero_grad()

        pred = model(src, src_lens)  # predict next step, init hidden state to zero at the begining of the sequence
        #output = [BATCH_SIZE, N, OUT_SIZE]

        loss = criterion(pred[:, :,0:2].contiguous().view(-1, 2), 
                         ((batch['trg'][:, :, :].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
                        ).mean()

        # print(pred.shape)

        # print(batch['trg'].shape)

        # print(criterion(pred[:, :,0:2].contiguous().view(-1, 2), 
        #                  ((batch['trg'][:, :, :].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
        #                 ).shape)

        # print(loss)
        # sys.exit(0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def valid(model, iterator, criterion, mean, std, device, dataset, N):

    model.eval()
    with torch.no_grad():

        epoch_loss = 0

        for batch_idx, batch in enumerate(iterator):
            src=(batch['src'].to(device)-mean.to(device))/std.to(device)
            trg=(batch['trg'].to(device)-mean.to(device))/std.to(device)

            src_lens = batch['src_lens']
            trg_lens = batch['trg_lens']

            pred = model(src, src_lens)  # predict next step, init hidden state to zero at the begining of the sequence
            #output = [BATCH_SIZE, N, OUT_SIZE]

            loss = criterion(pred[:, :,0:2].contiguous().view(-1, 2), 
                            ((batch['trg'][:, :, :].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)
                            ).mean()

            epoch_loss += loss.item()

        epoch_loss2 = []
        for idx, trajectory in dataset.whole_2d().items():
            tmp = np.diff(trajectory[:,[0,1]], axis=0)
            for i in range(0,tmp.shape[0]-N*2+1):
                src=((torch.Tensor(tmp[i:i+N]).to(device)-mean.to(device))/std.to(device)).unsqueeze(0)
                trg=((torch.Tensor(tmp[i+N:i+N*2]).to(device)-mean.to(device))/std.to(device)).unsqueeze(0)

                src_lens = torch.Tensor([src.shape[1]])

                pred = model(src, src_lens)

                loss = criterion(pred[:, :,0:2].contiguous().view(-1, 2), 
                                trg.contiguous().view(-1, 2)
                                ).mean()
                epoch_loss2.append(loss.item())

    return epoch_loss / len(iterator), sum(epoch_loss2)/len(epoch_loss2)

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

            # TODO test missing values, fill dropped point x,y as 0 (pad) XXXXX
            #inp[1:-1:2, [0,1]] = 0

            out = predict.predict2d_BLSTM(inp, model, mean=mean, std=std, out_time=3.0, fps=dataset.fps(), touch_ground_stop=True, device=device)

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
    parser = argparse.ArgumentParser(description="BLSTM Training Program")
    parser.add_argument("-t","--time", type=float, help="Input Sequence Time", default=0.1)
    parser.add_argument("-e","--epoch", type=int, help="Training Epochs", default=200)
    parser.add_argument("--hidden_size", type=int, help="Hidden Size", default=128)
    parser.add_argument("--hidden_layer", type=int, help="Hidden Layer", default=2)
    parser.add_argument("--physics_data", type=int, help="Training Datas", default=200000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001) # Adam default lr=0.001
    parser.add_argument("--save_epoch", type=int, help="Save at each N epoch", default=1)
    parser.add_argument('--fig_pth', type=str, default='./figure/BLSTM/')
    parser.add_argument('--wgt_pth', type=str, default='./weight/BLSTM/')
    parser.add_argument('--noisexy', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.2151959552)
    parser.add_argument('--draw_inference', action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.fig_pth, exist_ok=True)
    os.makedirs(args.wgt_pth, exist_ok=True)

    #N = args.seq # Time Sequence Number
    N_EPOCHS = args.epoch
    HIDDEN_SIZE = args.hidden_size
    N_LAYERS = args.hidden_layer
    N_PHYSICS_DATA = args.physics_data
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    SAVE_EPOCH = args.save_epoch

    print(f"Input Time {args.time}\n"
          f"Epoch: {N_EPOCHS}\n"
          f"Hidden Size: {HIDDEN_SIZE}\n"
          f"Hidden Layer: {N_LAYERS}\n"
          f"Physics data: {N_PHYSICS_DATA}\n"
          f"Batch Size: {BATCH_SIZE}\n"
          f"Learning Rate: {LEARNING_RATE}\n"
          f"Save At Each {SAVE_EPOCH} epoch\n")

    IN_SIZE = 2
    OUT_SIZE = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train Dataset
    train_dataset = PhysicsDataSet(datas=N_PHYSICS_DATA, model='BLSTM', 
                                   in_max_time=args.time, out_max_time=args.time*2, fps_range = (120.0,120.0), 
                                   elevation_range = (-89.9,89.9), speed_range = (1.0,240.0), 
                                   output_fps_range = (120.0,120.0), noise_xy=args.noisexy, 
                                   dxyt=True, network_in_dim=2, drop_mode=0, alpha=args.alpha)
    # print("DEBUG: train dataset arange")
    # train_dataset = PhysicsDataSet(model='BLSTM', 
    #                                in_max_time=args.time, out_max_time=args.time*2, fps_range = (120.0,120.0), 
    #                                elevation_range = (-89.9,89.9), speed_range = (1.0,240.0), 
    #                                output_fps_range = (120.0,120.0), noise_xy=args.noisexy, 
    #                                dxyt=True, network_in_dim=2, drop_mode=0, random='arange', alpha=args.alpha)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)

    # Valid
    # valid_dataset = PhysicsDataSet(in_max_time=args.time, out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')

    # vl_dl = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = True, drop_last=True)


    # Inference Dataset
    infer_dataset = RNNDataSet(dataset_path="../trajectories_dataset/", fps=120, smooth_2d=True)
    # print("Inference Last Experiment (Overfitting)")
    # infer_dataset = PhysicsDataSet(in_max_time=args.time, out_max_time=10.0, cut_under_ground=True, alpha=args.alpha, random='experiment', model='BLSTM')

    model = Blstm(in_size=IN_SIZE, out_size=OUT_SIZE, hidden_size=HIDDEN_SIZE, hidden_layer=N_LAYERS, device=device).to(device)
    print(f'The model has {param.count_parameters(model):,} trainable parameters')
    model.apply(param.init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # criterion = nn.MSELoss(reduction='mean')
    criterion = F.pairwise_distance

    mean = train_dataset.mean()
    std = train_dataset.std()

    hist_tr_loss = []

    print(f"Start Training ... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    for epoch in range(1,N_EPOCHS+1):

        tr_loss = train(model, tr_dl, optimizer, criterion, mean=mean, std=std, device=device)
        hist_tr_loss.append(tr_loss)

        # vl_loss, vl_loss2 = valid(model, vl_dl, criterion, mean=mean, std=std, device=device, dataset=valid_dataset, N=int(valid_dataset.fps()*args.time))

        infer_loss = inference(model, infer_dataset, mean=mean, std=std, device=device, epoch=epoch, N=math.floor(infer_dataset.fps()*args.time)+1)

        print(f"Epoch: {epoch}/{args.epoch}. Train Loss: {tr_loss:.8f}. Infer Loss: {infer_loss:.8f}. ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        if epoch % SAVE_EPOCH == 0:
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'datas':args.physics_data,
                        'mean':mean,
                        'std':std,
                        'hidden_size':args.hidden_size,
                        'hidden_layer':args.hidden_layer,
                       },  os.path.join(args.wgt_pth, f'BLSTM_p{args.physics_data}_e{epoch}'))
            # print(f"Save Weight At Epoch {epoch}")

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'datas':args.physics_data,
        'mean':mean,
        'std':std
        },  os.path.join(args.wgt_pth, f'BLSTM_final'))


    fig, ax = plt.subplots()
    ax.plot(range(1, len(hist_tr_loss)+1) , hist_tr_loss)
    ax.set_title('BLSTM')
    ax.set_ylabel('Train Loss')
    ax.set_xlabel('Epoch')
    plt.show()
