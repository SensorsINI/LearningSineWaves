# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:06:12 2020

@author: Marcin
"""

import torch
import torch.nn as nn
from torch.utils import data

import numpy as np
# from tqdm import tqdm
from tqdm.notebook import tqdm
import collections
from time import sleep


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


class Sequence(nn.Module):
    def __init__(self, args):
        super(Sequence, self).__init__()
        self.device = get_device()
        self.args = args
        self.gru1 = nn.GRUCell(1, args.h1_size).to(self.device)
        self.gru2 = nn.GRUCell(args.h1_size, args.h2_size).to(self.device)
        self.linear = nn.Linear(args.h2_size, 1).to(self.device)
        
        self.h_t = None
        self.h_t2 = None
        self.output = None
        self.outputs = []

        self.to(self.device)


    def initialize_sequence(self, input):
            starting_input = input[:,:self.args.warm_up_len]
            self.h_t = torch.zeros(starting_input.size(0), self.args.h1_size, dtype=torch.float).to(self.device)
            self.h_t2 = torch.zeros(starting_input.size(0),self.args.h2_size, dtype=torch.float).to(self.device)
            for i, input_t in enumerate(starting_input.chunk(starting_input.size(1), dim=1)):
                self.h_t = self.gru1(input_t, self.h_t)
                self.h_t2 = self.gru2(self.h_t, self.h_t2)
                self.output = self.linear(self.h_t2)
                self.outputs += [self.output]

    def forward(self,
                predict_len = 0,
                terminate = False,
                input = None,
                one_point_only = False):
        
        if one_point_only:
            if input is None:
                print('You need intput in forward function for this type of training')
            training_input = input[:,self.args.warm_up_len:self.args.warm_up_len+predict_len]
            for i, input_t in enumerate(training_input.chunk(training_input.size(1), dim=1)):
                if i>=predict_len:
                    break
                self.h_t = self.gru1(input_t, self.h_t)
                self.h_t2 = self.gru2(self.h_t, self.h_t2)
                self.output = self.linear(self.h_t2)
                self.outputs += [self.output]
        else:
            for i in range(predict_len):# if we should predict the future
                self.h_t = self.gru1(self.output, self.h_t)
                self.h_t2 = self.gru2(self.h_t, self.h_t2)
                self.output = self.linear(self.h_t2)
                # output_np = self.output.detach().cpu().numpy()
                self.outputs += [self.output]
                #print(i)
            
        if terminate:
            self.outputs = torch.stack(self.outputs, 1).squeeze(2)
            #print(self.outputs.is_cuda)
            return self.outputs
    
    def reset(self):
        self.h_t = None
        self.h_t2 = None
        self.output = None
        self.outputs = []


def load_model(net, savepath):
    pre_trained_model = torch.load(savepath, map_location=torch.device('cpu'))
    print("Loading Model: ", savepath)
    
    pre_trained_model = list(pre_trained_model.items())
    new_state_dict = collections.OrderedDict()
    count = 0
    num_param_key = len(pre_trained_model)
    for key, value in net.state_dict().items():
        if count >= num_param_key:
            break
        layer_name, weights = pre_trained_model[count]
        new_state_dict[key] = weights
        print("Pre-trained Layer: %s - Loaded into new layer: %s" % (layer_name, key))
        count += 1
    net.load_state_dict(new_state_dict)


def print_parameter_count(net):
    params = 0
    for param in list(net.parameters()):
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        params += sizes
    print('::: # network parameters: ' + str(params))

def initialize_net_parameters(net):
    for name, param in net.named_parameters():
        print(name)
        if 'gru' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
        if 'linear' in name:
            if 'weight' in name:
                nn.init.orthogonal_(param)
                # nn.init.xavier_uniform_(param)
        if 'bias' in name:  # all biases
            nn.init.constant_(param, 0)


def train_net(net, train_generator, criterion, optimizer, args, epoch):
    net = net.train()
    train_loss = 0
    train_batches = 0

    if epoch%2 == 0:
        # Global view
        PL =  (epoch+1)*args.window
        print('In this epoch we train global perspective.')
    elif epoch%2 == 1:
        # Local view
        PL =  epoch*args.window
        print('In this epoch we train local perspective.')

    for batch, labels in tqdm(train_generator):  # Iterate through batches
        # Move data to GPU
        if torch.cuda.is_available():
            batch = batch.float().cuda().squeeze()
            labels = labels.float().cuda().squeeze()
            
        else:
            batch = batch.float().squeeze()
            labels = labels.float().squeeze()
            
        # np_batch = batch.numpy()
        # np_label = labels.numpy()
        
        net.initialize_sequence(batch)



        # Predict future points
        if epoch%2 == 0:
            out = net(predict_len = args.warm_up_len,
            terminate = False
            )
            # Adjust size of labels tensor 
            labels  = labels[:,:2*args.warm_up_len+PL]
            # Optimization
            optimizer.zero_grad()
            # Global view
            out = net(predict_len = PL,
                        terminate = True
                        )
        elif epoch%2 == 1:
            # Adjust size of labels tensor 
            labels  = labels[:,:args.warm_up_len+PL]
            # Optimization
            optimizer.zero_grad()
            # Local view
            out = net(  predict_len = PL,
                        terminate = True,
                        input = batch,
                        one_point_only = True,
                        )
            


        # Get loss
        loss = criterion(out[:,args.warm_up_len:], labels[:,args.warm_up_len:])

        # Backward propagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

        # Update parameters
        optimizer.step()

        # Increment monitoring variables
        batch_loss = loss.detach()
        train_loss += batch_loss  # Accumulate loss2
        train_batches += list(batch.shape)[0]  # Accumulate count so we can calculate mean later
        
        net.reset()

    train_loss_normed = train_loss.detach().cpu().numpy() / train_batches   #TODO maybe changing it to cpu() here messes the next iteration

    return train_loss_normed


def validate_net(net, dev_generator, criterion, args):
    net = net.eval()
    net.reset()
    dev_loss = 0
    dev_batches = 0

    if (args.num_epochs-1)%2 == 0:
        PL =  ((args.num_epochs-1)+1)*args.window
    elif (args.num_epochs-1)%2 == 1:
        PL =  (args.num_epochs-1)*args.window

    for (batch, labels) in tqdm(dev_generator):
        with torch.no_grad():
            
            if torch.cuda.is_available():
                batch = batch.float().cuda().squeeze()
                labels = labels.float().cuda().squeeze()
            else:
                batch = batch.float().squeeze()
                labels = labels.float().squeeze()

            # np_batch = batch.numpy()
            # np_label = labels.numpy()
                
            labels  = labels[:,:args.warm_up_len+PL]
            
            # Global view
            net.initialize_sequence(batch)
            out = net(predict_len = PL,
                        terminate = True
                        )
            #out_global  =  out.detach().cpu().numpy()
            loss_global = criterion(out[:,args.warm_up_len:], labels[:,args.warm_up_len:])
            net.reset()
            
        
            
            # Local view
            net.initialize_sequence(batch)
            out = net(  predict_len = PL,
                        terminate = True,
                        input = batch,
                        one_point_only = True,
                        )
            #out_local  =  out.detach().cpu().numpy()
            loss_local = criterion(out[:,args.warm_up_len:], labels[:,args.warm_up_len:])


            loss = loss_local+loss_global

            # Increment monitoring variables
            batch_loss = loss.detach()
            dev_loss += batch_loss
            dev_batches += list(batch.shape)[0]  # Accumulate count so we can calculate mean later
            
            net.reset()

    dev_loss_normed = dev_loss.detach().cpu().numpy() / dev_batches

    return dev_loss_normed



    

class Dataset(data.Dataset):
    def __init__(self, args):
        #'Initialization'
        
        
        self.T_min     = args.T_min
        self.T_max     = args.T_max
        self.phi_min   = args.phi_min
        self.phi_max   = args.phi_max
        self.A_min     = args.A_min
        self.A_max     = args.A_max
        self.dt        = args.dt
        self.epoch_len = args.epoch_len
        self.num_epochs = args.num_epochs
        self.warm_up_len = args.warm_up_len
        self.window = args.window

    def __len__(self):

        #'Total number of samples'
        return int(self.epoch_len)
    
    def __getitem__(self, idx, a = None, phi = None, period = None):


        sample_length = self.window*self.num_epochs+self.warm_up_len +1
        
        if a:
            A = a
        else:
            A   = np.random.uniform(low = self.A_min, high = self.A_max)
            
        if phi:
            Phi = phi
        else:
            Phi = np.random.uniform(low = self.phi_min, high = self.phi_max)
        
        if period:
            f = 1.0/period
        else:
            f   = np.random.uniform(low = 1.0/self.T_max, high = 1.0/self.T_min)
        
        

        t = np.arange(0, sample_length)*self.dt


        
        y = A*np.sin(t*f*2*np.pi+Phi)
        
        
        features = torch.from_numpy(y[:-1]).float().unsqueeze(1)
        targets = torch.from_numpy(y[1:]).float().unsqueeze(1)
        

        return features, targets
    
    


# This is the set just for validation
# The idea is to ensure the same data for each epoch and so to make validations better comparable
# Additionally the validation set is smaller to spead up the process
class DatasetValid(data.Dataset):
    def __init__(self, args):
        #'Initialization'
        
        # How many different parameters should be tested
        NA = 16
        Nf = 16
        NPhi = 16

        self.N_samples = NA*Nf*NPhi
        
        T_min     = args.T_min
        T_max     = args.T_max
        phi_min   = args.phi_min
        phi_max   = args.phi_max
        A_min     = args.A_min
        A_max     = args.A_max
        self.dt        = args.dt

        dA = (A_max-A_min)/(NA-1)
        dT = (T_max-T_min)/(Nf-1)
        dPhi = (phi_max-phi_min)/(NPhi-1)

        self.num_epochs = args.num_epochs
        self.warm_up_len = args.warm_up_len
        self.window = args.window



        self.parameters_table = np.zeros((self.N_samples, 3))
        
        idx = 0
        for i in range(NA):
            for j in range(Nf):
                for k in range(NPhi):
                    self.parameters_table[idx,:]   = (A_min   + i*dA, T_min   + j*dT, phi_min + k*dPhi)
                    idx +=1

        

    def __len__(self):

        #'Total number of samples'
        return int(self.N_samples)
    
    def __getitem__(self, idx, a = None, phi = None, period = None):
        
        sample_length = self.window*self.num_epochs+self.warm_up_len +1
        
        t = np.arange(0, sample_length)*self.dt

        (A, f, Phi) = self.parameters_table[idx,:]
        
        y = A*np.sin(t*f*2*np.pi+Phi)
        
        
        features = torch.from_numpy(y[:-1]).float().unsqueeze(1)
        targets = torch.from_numpy(y[1:]).float().unsqueeze(1)
        

        return features, targets
    

      
