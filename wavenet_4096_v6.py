#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.cuda.is_available()


# In[2]:


from pathlib import Path
import random

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
# These transformations will be passed to our model class
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from tqdm.auto import tqdm
import glob
from torch.distributions import Beta
from torchvision.ops import sigmoid_focal_loss


# In[3]:


from joblib import Parallel, delayed
import os
from os.path import exists

WAV_SIZE=2000
STEP_SIZE=500
TIMES_REAL=4
TIMES_TRAIN=8
is_mixed_precision = True
INPUT_PATH = '../input'

class GaitDataset(torch.utils.data.Dataset):

    def __init__(self, df, is_train=False,transforms=None):
        self.is_train = is_train
        self.data = df

    def __len__(self):
        if self.is_train:
            return len(self.data)*TIMES_TRAIN
        else:
            return len(self.data)
    
    
    def __getitem__(self, idx):
        if self.is_train:
            idx = np.random.randint(0,len(self.data))
            
        row = self.data.iloc[idx]
        wid = row.Id
        subject = row.Subject
        t = row.type
        
        if t == 0:
            wav = np.load(f'{INPUT_PATH}/train/tdcsfog_np/{row.Id}_sig.npy')
            tgt = np.load(f'{INPUT_PATH}/train/tdcsfog_np/{row.Id}_tgt.npy')
        else:
            wav = np.load(f'{INPUT_PATH}/train/defog_np//{row.Id}_sig.npy')
            tgt = np.load(f'{INPUT_PATH}/train/defog_np/{row.Id}_tgt.npy')
            
        
        wav = wav/40.
        
        label = tgt
        wav_df = pd.DataFrame(wav)
        tgt_df = pd.DataFrame(label)
        
        wavs = []
        tgts = []
        if self.is_train:
            for w in wav_df.rolling(WAV_SIZE,step=STEP_SIZE):
                if w.shape[0] == WAV_SIZE:
                    wavs.append(w.values)

            if len(wavs) ==0:
                wavs = [wav]

            for w in tgt_df.rolling(WAV_SIZE,step=STEP_SIZE):
                if w.shape[0] == WAV_SIZE:
                    tgts.append(w.values)

            if len(tgts) ==0:
                tgts = [label]
                
            wav = np.stack(wavs,axis=0)
            label = np.stack(tgts,axis=0)
            actual_len=-1
        
        else:
            actual_len = len(wav)
            nchunk = (len(wav)//WAV_SIZE)+1
            wav = wav.reshape(-1,len(wav),3)
            label = label.reshape(-1,len(label),3)
            
        
        if self.is_train and len(wav)>1:
            if row.type == 0:
                rix = np.random.randint(0,len(wav))
                wav = wav[rix:rix+1]
                label = label[rix:rix+1]
            else:
                rix = np.random.randint(0,len(wav),TIMES_REAL)
                wav = wav[rix]
                label = label[rix]
        
        #print('wav',wav.shape, label.shape)
        
        sample = {"wav": wav, "label":label, "actual_len":actual_len}
        
        #print('label',label.shape,tgt.shape)
        #print('wav',wav.shape)

        return sample
        
def collate_wrapper(batch):
    out = {}
    wavs = []
    labels = []
    s_ix1s = []
    e_ix1s = []
    for item in batch:
        wavs.append(item['wav'])
        labels.append(item['label'])
        
    out['wav'] = torch.from_numpy(np.concatenate(wavs,axis=0))
    out['label'] = torch.from_numpy(np.concatenate(labels,axis=0))
    
    return out

def getDataLoader(params,train_x,val_x,train_transforms=None,val_transforms=None):
    
    train_dataset = GaitDataset(
            df=train_x, is_train=True, transforms=train_transforms
        )
    val_dataset = GaitDataset(df=val_x, transforms=val_transforms)
    
    trainDataLoader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=params['batch_size'],
                            num_workers=params['num_workers'],
                            shuffle=True,collate_fn = collate_wrapper,
                            pin_memory=False,
                            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id)
                        )
    valDataLoader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=1,
                        num_workers=params['num_workers'],
                        shuffle=False,
                        pin_memory=False,
                    )
    
    return trainDataLoader,valDataLoader


# In[4]:


class Mixup(nn.Module):
    def __init__(self, mix_beta=1):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1,1) * Y + (1 - coeffs.view(-1, 1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight


# In[5]:


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
# detail 
class Classifier(nn.Module):
    def __init__(self, inch=3, kernel_size=3):
        super().__init__()
        self.LSTM = nn.GRU(input_size=128, hidden_size=128, num_layers=4, 
                           batch_first=True, bidirectional=True)
        
        #self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(inch, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc1 = nn.Linear(256, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        #x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, h = self.LSTM(x)
        x = self.fc1(x)
    
        
        return x,x,x


# In[6]:


import torch.cuda.amp as amp
class AmpNet(Classifier):
    
    def __init__(self,params):
        super(AmpNet, self).__init__()
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)

is_mixed_precision = True  #True #False


# In[7]:


def getOptimzersScheduler(model,params,steps_in_epoch=25,pct_start=0.1):
    
    
    mdl_parameters = [
                {'params': model.parameters(), 'lr': 1e-4},
                #{'params': model.fc.parameters(), 'lr': 1e-4},
                #{'params': model.attention.parameters(), 'lr': 1e-5},
            ]
    
    optimizer = torch.optim.Adam(mdl_parameters, lr=params['learning_rate'][0])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,steps_per_epoch=1,
                                                    pct_start=pct_start,
                                                    max_lr=params['learning_rate'],
                                                    epochs  = params['max_epochs'], 
                                                    div_factor = params['div_factor'], 
                                                    final_div_factor=params['final_div_factor'],
                                                    verbose=True)
    
    return optimizer,scheduler,False


# In[8]:


def save_model(epoch,model,ckpt_path='./',name='',val_rmse=0):
    path = os.path.join(ckpt_path, '{}_{}.pth'.format(name, epoch))
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)
    
def load_model(model,ckpt_path):
    state = torch.load(ckpt_path)
    print(model.load_state_dict(state,strict=False))
    return model


# In[9]:


def focal_loss(pred,target):
    return 32*sigmoid_focal_loss(pred,target,reduction='mean')


# In[10]:


def training_step(model, batch, batch_idx,optimizer,scheduler,isStepScheduler=False):
    # Load images and labels
    x = batch["wav"].float()
    y = batch["label"].float()
    
   
    mixup = Mixup()
    
    ##Mixup Aug
    if np.random.uniform(0,1) < 0.:
        x,y = mixup(x,y)
    
    #print('x',x.shape,ys1.shape,ye1.shape)
    
    if GPU:
        x, y  = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    criterion = focal_loss #torch.nn.BCEWithLogitsLoss(reduction="mean") 
    reg_criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = FocalLoss()
    
    #optimizer.zero_grad()
    iters_to_accumulate=2
    # Forward 

    if is_mixed_precision:
        with amp.autocast():
            preds, sp, ep = model(x)
            b,s,c = y.shape
            y = y.reshape(b*s,c)
            preds = preds.reshape(b*s,-1)
            loss = criterion(preds,y)/ iters_to_accumulate
            
            #print('pred',sp.shape,ys1.shape, ep.shape)
            #rloss1 = reg_criterion(sp,ys1)
            #rloss2 = reg_criterion(ep,ye1)
            #loss = loss + rloss1 + rloss2
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % iters_to_accumulate == 0:
                #print('accumulating')
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            
            loss = loss.item()
    else:
        preds = model(x,att_mask)
        loss = criterion(preds.flatten(), y.flatten())
        loss.backward()
        optimizer.step()
        loss = loss.item()
        
    if isStepScheduler:
        scheduler.step()

    # Calculate validation IOU (global)
    #preds = preds.detach()
    #y = y.detach().cpu()
    return loss

def validation_step(model, batch, batch_idx):
    # Load images and labels
    x = batch["wav"].float()
    y = batch["label"].float()
    actual_len = batch['actual_len'].long()
    iters_to_accumulate= 2
    
    if GPU:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    criterion = focal_loss #torch.nn.BCEWithLogitsLoss(reduction="mean") 
    #criterion = FocalLoss()

    x = x[0]
    y =y[0]
    actual_len=actual_len[0]
    
    BS=20
    
    preds_list = []
    tgt_list = []

    # Forward pass & softmax
    with torch.no_grad():
        if is_mixed_precision:
            with amp.autocast():
                num_iter =  x.shape[0]//BS
                if num_iter== 0:
                    num_iter=1
                for b in range(num_iter):
                    preds,_,_ = model(x[b*BS:(b+1)*BS])
                    yb = y[b*BS:(b+1)*BS]
                    b,s,c = yb.shape
                    yb = yb.reshape(b*s,c)
                    preds = preds.reshape(b*s,-1)
                    preds_list.append(preds)
                    tgt_list.append(yb)
                   
                preds = torch.cat(preds_list,dim=0)
                y = torch.cat(tgt_list,dim=0)
                
                #print('preds',preds.shape,y.shape)
                
                y = y[0:actual_len]
                preds = preds[0:actual_len]
                loss = criterion(preds, y)/ iters_to_accumulate
                
    preds = torch.sigmoid(preds)
    
    loss = loss.item()
    return loss,preds.detach().cpu().numpy(),y.long().detach().cpu().numpy()


# In[11]:


from sklearn.metrics import roc_auc_score,f1_score,precision_score,average_precision_score
def train_epoch(model,trainDataLoader,optimizer,scheduler,isStepScheduler=True):
    total_intersection=0
    total_union=0
    total_loss=0
    model.train()
    torch.set_grad_enabled(True)
    total_step=0
    ious = []
    
    
    pbar = tqdm(enumerate(trainDataLoader),total=len(trainDataLoader))
    for bi,data in pbar:
        loss= training_step(model,data,bi,optimizer,scheduler)
        total_loss+=loss
        total_step+=1
        pbar.set_postfix({'loss':total_loss/total_step})
        
    if not isStepScheduler: #in case epoch based scheduler
        scheduler.step()
            
    total_loss /= total_step
    return total_loss
        

def val_epoch(model,valDataLoader):
    total_intersection=0
    total_union=0
    total_loss=0
    
    total_step=0
    model.eval()
    preds = []
    targets = []
    pbar=tqdm(enumerate(valDataLoader),total=len(valDataLoader))
    for bi,data in pbar :
        loss, pred ,tgt = validation_step(model,data,bi)
        total_loss+=loss
        total_step+=1
        preds.extend(pred)
        targets.extend(tgt)
        
        pbar.set_postfix({'loss':total_loss/total_step})
        
    preds = np.stack(preds)
    preds = np.clip(preds,0,1)
    targets = np.stack(targets)
    
    #preds = preds[targets!=0]
    #targets = targets[targets!=0]
    
    print('targets',targets.shape, preds.shape)
    aps = []
    for i in range(3):
        score = average_precision_score(targets[:,i],preds[:,i])
        aps.append(score) 
    
    APx = average_precision_score(targets,preds,average='macro')
    AP = np.mean(aps)
    
    del targets,preds
    gc.collect()
    
    print('AP', AP, APx)
    total_loss /= total_step
    return total_loss,AP


# In[12]:


GPU=True
def training_loop(params,train_x,val_x,savedir='./',mdl_name='resnet34'):
    
    #create model
    model = AmpNet(params).cuda()
    #load model
    
    #get loaders
    train_transforms=None
    val_transforms = None
    trainDataLoader,valDataLoader = getDataLoader(params,train_x,val_x,train_transforms,val_transforms)
    
    optimizer,scheduler,isStepScheduler = getOptimzersScheduler(model,params,
                                                                steps_in_epoch=len(trainDataLoader),
                                                                pct_start=0.1)
    best_ap= 0
    #control loop
    for e in range(params['max_epochs']):
        train_loss = train_epoch(model,trainDataLoader,optimizer,scheduler,isStepScheduler)
        loss, AP = val_epoch(model,valDataLoader)
        #logging here
        #print(e,'Train Result',f'loss={train_loss}')
        print(e,'Val Result',f'AP={AP} ')
        if AP > best_ap :
            print(f'Saving for AP {AP}')
            save_model(e,model,ckpt_path=savedir,name=mdl_name,val_rmse=best_ap)
            best_ap=AP
        else:
            print(f'Not Saving for AP {AP}')
        


# In[13]:


hparams = {
    # Optional hparams
    "backbone": 'wavenet_4096', #'', #'tf_efficientnetv2_b2',
    "learning_rate": [5e-4],
    "max_epochs": 71,
    "batch_size": 16,
    "num_workers": 0,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": f"",
    "gpu": torch.cuda.is_available(),
    'div_factor':5,
    'final_div_factor':10,
}


# In[14]:


scaler = amp.GradScaler()


# In[15]:


import random
seed=42
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms = True
    random.seed(0)
    np.random.seed(0)
set_seed(seed)



# In[16]:


events = pd.read_csv(f'{INPUT_PATH}/events.csv')
events = events[~events.Type.isnull()]

defog = pd.read_csv(f'{INPUT_PATH}/defog_metadata.csv')
defog = defog[defog.Id.isin(events.Id)].reset_index(drop=True)

tdcsfog = pd.read_csv(f'{INPUT_PATH}/tdcsfog_metadata.csv')
tdcsfog = tdcsfog[tdcsfog.Id.isin(events.Id)].reset_index(drop=True)

defog['type'] = 1
tdcsfog['type'] = 0

train = pd.concat([tdcsfog]).reset_index(drop=True)


# In[17]:


set(tdcsfog.Subject.unique()).intersection(set(defog.Subject.unique()))


# In[18]:


from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold,GroupKFold
#kf = GroupKFold(n_splits=5,random_state=42,shuffle=True)
kf = GroupKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(kf.split(tdcsfog.Id,tdcsfog.Medication,groups=tdcsfog.Subject)):
    tdcsfog.loc[test_index,'fold'] =i
    
#kf = GroupKFold(n_splits=5,random_state=42,shuffle=True)
kf = GroupKFold(n_splits=5)
for i, (train_index, test_index) in enumerate(kf.split(defog.Id,defog.Medication,groups=defog.Subject)):
    defog.loc[test_index,'fold'] =i


# In[ ]:


import gc
version='6'
fn=0
for fn in [1,4,0,2,3]:  
    set_seed()

    mdl_name=hparams['backbone']
    savedir = f'trained-models-{mdl_name}-v{version}'
    Path(savedir).mkdir(exist_ok=True, parents=True)
    
    val = pd.concat([defog[defog.fold==fn],tdcsfog[tdcsfog.fold==fn]])
    tr = pd.concat([defog[defog.fold!=fn],tdcsfog[tdcsfog.fold!=fn]])
    
    print('Train',tr.shape,'Val',val.shape)
    
   
    training_loop(hparams,tr,val,savedir=savedir,mdl_name=f'{mdl_name}-fold{fn}')
    gc.collect()

    #break


# In[ ]:




