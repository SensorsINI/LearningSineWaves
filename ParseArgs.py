import argparse
import numpy as np


savepath = './save/'+ 'MyNet' + '.pt'
savepathPre = './save/' + 'MyNetPre' + '.pt'

def args():
    parser = argparse.ArgumentParser(description='Train a GRU network.')

    parser.add_argument('--T_min',          default=10.0/1.0e3,      type=float,  help='Minimum allowed period of sine wave in s')
    parser.add_argument('--T_max',          default=200.0/1.0e3,     type=float,  help='Maximum allowed period of sine wave in s') # Make it 200
    parser.add_argument('--phi_min',        default=-np.pi/2,        type=float,  help='Minimum allowed phase shift of sine wave')
    parser.add_argument('--phi_max',        default=+np.pi/2,        type=float,  help='Maximum allowed phase shift of sine wave')
    parser.add_argument('--A_min',          default=0.1,             type=float,  help='Minimum allowed amplitude of sine wave')
    parser.add_argument('--A_max',          default=1.0,             type=float,  help='Maximum allowed amplitude of sine wave, let it less equal 1 - there is no normalization of the data applied in the program')

    parser.add_argument('--dt',             default=1.0/1.0e3,       type=float,  help='Time interval of a time step in s')
    parser.add_argument('--warm_up_len',    default=500,         type=int,    help='Number of timesteps feed into NN during training')  # TODO make it 500
    parser.add_argument('--exp_len',        default=1500,          type=int,    help='Number of timesteps feed into NN during training') # TODO make it 10000

    parser.add_argument('--h1_size',        default=256,             type=int,    help='First hidden layer size')
    parser.add_argument('--h2_size',        default=256,             type=int,    help='Second hindden layer size')
    
    parser.add_argument('--lr',             default=1.0e-4,          type=float,  help='Learning rate')
    parser.add_argument('--batch_size',     default=32,              type=int,    help='Size of a batch')
    parser.add_argument('--num_epochs',     default=10,              type=int,    help='Number of epochs of training')
    parser.add_argument('--epoch_len',      default=3e5,             type=int,    help='How many sine waves are fed in NN during one epoch of training')
    parser.add_argument('--window',         default=100,              type=int,    help='Window of interatively predicted future')
    parser.add_argument('--pretrained',     default=True,              type=bool,    help='Say if one should load pretrained net')
    

    parser.add_argument('--savepath',       default=savepath,        type=str,    help='Number of workers to produce data from data loaders')
    parser.add_argument('--savepathPre',    default=savepathPre,     type=str,    help='Path from where to load a pretrained model')

    args = parser.parse_args()
    return args  