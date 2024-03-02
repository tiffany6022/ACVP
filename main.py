import math
import os

import numpy as np
import torch
import torch.nn as nn

from import_these import *


def training(model, opt, dl, epoch, result_path):
    model.train()
    losses = []

    for x, y in dl:
        _, loss = model(x)
        losses.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()

    print(f'{epoch}: loss:{np.mean(losses):.6f}')


if __name__ == '__main__':

    main_dir = '/home/tiffany/git/iACVP_alldata/data/dataset'
    batch_size = 32
    lr = 0.001
    epoch = 100000
    result_path = f'result'
    os.makedirs(f'{result_path}/model_ckpt/', exist_ok=True)

    # Fixing the random seed
    torch_seed = 0
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the ProteinDataset for pretraining
    pretrain_dataset = ProteinDataset(
        [
            f'cross_val/1/cv_train_1.csv',
            f'cross_val/1/cv_val_1.csv',
            f'independent_test/independent_test.csv',
        ]
    )

    # Create a DataLoader for the pretraining dataset
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True
    )

    # Initialize the model
    device = torch.device('cuda:0')
    model = Model(hid_dims=[128, 256], device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(1, epoch+1):
        training(model, opt, pretrain_loader, i, result_path)
        if all(char == '0' for char in str(i)[1:]):
            torch.save(model, f'{result_path}/model_ckpt/checkpoint_{i}.ckpt')
