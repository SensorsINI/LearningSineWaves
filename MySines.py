# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:35:39 2020

@author: Marcin
"""
import torch

import os
import time
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.utils.data.dataloader



import numpy as np

from memory_profiler import profile
import timeit

from utilis_tests import  plots

import sys, importlib

import utilis
importlib.reload(sys.modules['utilis'])

import ParseArgs
importlib.reload(sys.modules['ParseArgs'])

  
device = utilis.get_device()

args = ParseArgs.args()
print(args.__dict__)

    
#@profile(precision=4)
def train_network():

    start = timeit.default_timer()

    # Make folders
    try:
        os.makedirs('save')
    except:
        pass


    # Create PyTorch Dataset
    train_set = utilis.Dataset(args)
    dev_set = utilis.DatasetValid(args)

    # Create PyTorch dataloaders for train and dev set
    train_generator = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False)
    dev_generator = data.DataLoader(dataset=dev_set, batch_size=512, shuffle=False)

    net = utilis.Sequence(args)

    if args.pretrained:

        # If a pretrainded model exists load the parameters
        print('Use pretrained network')
        utilis.load_model(net, args.savepathPre)

    else:
        print('Initialize new network')
        # Initialize weights and biases if no pretrained model is loaded
        utilis.initialize_net_parameters(net)


    # Print parameter count
    utilis.print_parameter_count(net)

    # Select Optimizer
    optimizer = optim.Adam(net.parameters(), amsgrad=True, lr=args.lr)

    # Select Loss Function
    criterion = nn.MSELoss()  # Mean square error loss function

    # Initial validation of the pretrained net
    if args.pretrained:
        dev_loss_normed = utilis.validate_net(net, dev_generator, criterion, args)
        min_dev_loss = dev_loss_normed
        print('Dev_loss of the pretrained network: ' + str(min_dev_loss))
        plots(net, args)
        time.sleep(0.5)

    ########################################################
    # Training
    ########################################################


    # Epoch loop
    print("Starting training...")

    dev_gain = 1

    # Timer
    # start = timeit.default_timer()

    epoch_saved = -1
    for epoch in range(args.num_epochs):
        
        ###########################################################################################################
        # Training - Iterate batches
        ###########################################################################################################
        train_loss_normed = utilis.train_net(net, train_generator, criterion, optimizer, args, epoch)

        ###########################################################################################################
        # Validation - Iterate batches
        ###########################################################################################################
        dev_loss_normed = utilis.validate_net(net, dev_generator, criterion, args)



        # Get current learning rate
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

        if epoch == 0 and not args.pretrained:
            min_dev_loss = dev_loss_normed
        elif epoch == 0 and args.pretrained:
            pass # min_dev_loss already set

        # Get relative dev loss gain
        if epoch >= 1:
            dev_gain = (min_dev_loss - dev_loss_normed) / min_dev_loss


        print('\nEpoch: %3d of %3d | '
              'LR: %1.5f | '
              'Train-L: %6.4f | '
              'Val-L: %6.4f | '
              'Val-Gain: %3.2f |' % (epoch, args.num_epochs - 1,
                                  lr_curr,
                                  train_loss_normed,
                                  dev_loss_normed,
                                  dev_gain * 100))

        # Save the best model with the lowest dev loss
        if dev_loss_normed <= min_dev_loss:
            epoch_saved = epoch
            min_dev_loss = dev_loss_normed
            torch.save(net.state_dict(), args.savepath)
            print('>>> saving best model from epoch {}'.format(epoch))
            plots(net, args)
        else:
            print('>>> We keep model from epoch {}'.format(epoch_saved))
            plots(net, args)
            if args.pretrained and epoch_saved == -1: # Load a pretrained model again
                #utilis.load_model(net, args.savepathPre)
                pass
            elif not args.pretrained and epoch_saved == -1: # Reinicialize randomly
                # Initialize weights and biases if no pretrained model is loaded
                #utilis.initialize_net_parameters(net)
                pass
            elif epoch_saved > -1:
                pass
                #utilis.load_model(net, args.savepath)


    print("Training Completed...                                               ")
    print(" ")

    stop = timeit.default_timer() 
    total_time = stop-start
    
    return total_time




if __name__ == '__main__':
  total_time = train_network()
  print('Total time of training the network: '+str(total_time))