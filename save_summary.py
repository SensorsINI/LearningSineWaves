# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 06:34:03 2020

@author: Marcin
"""


from utilis_tests import all_results
from utilis import get_device, Sequence

import ParseArgs

import torch

import collections


args = ParseArgs.args()

print(args.__dict__)
  
device = get_device()



def save_summary():
    
    net = Sequence(args)
    
    # If a pretrainded model exists load the parameters
    pre_trained_model = torch.load(args.savepathPre, map_location=torch.device('cpu'))
    print("Loading Model: ", args.savepathPre)
    
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
    
    all_results(net, args)
    



if __name__ == '__main__':
    save_summary()


