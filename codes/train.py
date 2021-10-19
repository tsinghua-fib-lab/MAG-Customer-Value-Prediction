import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def test(model, test_loader,args):
    
    model.eval()
    loss, correct = 0., 0.
    data = test_loader['all']
    data = data.to(args.device)
    out = model(data)   
    y_truth = data.y[data.y_index]
    mae_loss = F.l1_loss(out, y_truth , reduction='mean').item()
    rmse_loss = F.mse_loss(out, y_truth, reduction='mean').item()
    rmse_loss =  np.sqrt(rmse_loss)
    
    return mae_loss, rmse_loss

def update_func(epoch):
    return 1
'''
    if epoch <= 90:
        return 1 
    elif epoch <= 1000:
        return 0.1
    else:
        return 0.01
'''

def train(model, train_loader, val_loader,test_loader, args):
    writer = SummaryWriter(log_dir='./LOG/model/{}-lr{}-wd{}-enh{}-gnh{}-nh{}-{}'.format(args.model, args.lr, args.weight_decay, args.enh, args.gnh, args.n_hidden, args.en))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=update_func)
    
    model.train()
    start = time.time()
    min_loss = 0
    patience = 0
    for epoch in range(args.epochs):
        print('Epoch {}:'.format(epoch))
        train_loss = 0
        for i in range(train_loader.batch_num):
            data = train_loader[i]
            data = data.to(args.device)
            optimizer.zero_grad()
            
            out = model(data)

            if args.loss_fn == 'mse':
                loss = F.mse_loss(out, data.y[data.batch_ids], reduction='sum')
            elif args.loss_fn == 'mae':
                loss = F.l1_loss(out, data.y[data.batch_ids], reduction='sum')
            elif args.loss_fn == 'huber':
                loss = F.smooth_l1_loss(out, data.y[data.batch_ids], reduction='sum')

            loss_mae = F.l1_loss(out, data.y[data.batch_ids], reduction='sum').item()
            train_loss += loss_mae
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
        train_loss = train_loss/train_loader.num
        print("Time {} -- Training loss:{}".format(time_iter, train_loss))
        writer.add_scalar('scalar/train/loss', train_loss, epoch)
        
        
        mae_loss, rmse_loss= test(model, val_loader,args)
        print("VAL:MAE:{} -- RMSE:{}".format(mae_loss, rmse_loss))
        val_mae_loss=mae_loss

        mae_loss, rmse_loss= test(model, test_loader,args)

        scheduler.step()
        
        if (epoch>100) and (val_mae_loss < float(args.early_stop)):
            break
        if patience > args.patience:
            break
    writer.close()

