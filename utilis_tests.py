# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:28:37 2020

@author: Marcin
"""


'''
args:
    dt = 0.001
    T_min = 0.01s    - 10 points per period
    

'''

import numpy as np
import torch

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


from tqdm import tqdm
import time

from utilis import get_device

from pylab import scatter



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def Calculate_A(data_series, args):
    
    t = np.arange(0,data_series.shape[0])*args.dt
    
    # Find peaks to calculate change of amplitude
    # What is a minimal distance between peaks which we can recognize given args?
    # This distance should is equal to the minimal wave period
    # The frequency may accelerate so the peaks are nearer
    # But probably I do not need to measure amplitude more often
    peaks, _ = find_peaks(data_series, distance=int(args.T_min/args.dt))
    
    # Find the data points corresponding to peaks
    t_peaks = t[peaks]
    y_peaks = data_series[peaks]
    
    # Append first and last element to do proper interpolation
    t_peaks = np.append(t_peaks, t[-1])
    t_peaks = np.insert(t_peaks, 0, 0.0)
    
    y_peaks = np.append(y_peaks, y_peaks[-1])
    y_peaks = np.insert(y_peaks, 0, y_peaks[0])
    
    fA = interp1d(t_peaks, y_peaks, kind='cubic')
    
    A = fA(t)
    A[A<args.A_min/2.0] = 0.0
    
    return A
    
    
    
    
def Calculate_f(data_series, args, A):
    
    t = np.arange(0,data_series.shape[0])*args.dt
    
    # f_min = 1.0/args.T_max # the smallest freq we are interested in [Hz]
    # f_max = 1.0/args.T_min # the biggest frequency we are interested in [Hz]
    # dt = args.dt # time step [s]
    
    # res = min(f_min/2.0, 1.0)
    
    # F_max = 1/(2*dt) # Biggest frequeny we can register
    
    # # F_max/N = res
    # N = F_max/res # Number of points our fourier transform must have
    

    
    condition_1 = (np.diff(np.sign(data_series))!=0)
    # Find zero-crossing to calculate change of period
    zero_crossings_idx = np.where(condition_1)[0]
    
    t_crossings = t[zero_crossings_idx]
    if len(t_crossings)<2:
        print('To few X-Axis crossings to calculate frequency')
        return None
    
    # Refine the point where the sines are crossing X-axis
    for i in range(len(zero_crossings_idx)):
        DT = args.dt*abs(data_series[zero_crossings_idx[i]])/(abs(data_series[zero_crossings_idx[i]])+abs(data_series[zero_crossings_idx[i]+1]))
        t_crossings[i] = t_crossings[i]+DT
        
    # Calculate period and find the corresponding point in time
    # (Between the time points used to calculate the frequency)
    Diff = np.diff(t_crossings)
    period_pred = 2*Diff
    t_period = t_crossings[:-1]+(Diff/2.0)
    

    # Smooth period
    # The experience showed that usually the negative and positive part of sine wave
    # Have slightly different frequeny. We calculate the mean to get rid of this oscillation
    period_pred = moving_average(period_pred, 2)
    t_period = moving_average(t_period, 2)
    
    #TODO: What is a maximal frequence I can register given args?
    condition_2 = (period_pred<args.T_min/2.0)|((period_pred>args.T_max*2.0))
    idx = np.where(condition_2)

    y_period = np.delete(period_pred, idx)
    t_period = np.delete(t_period, idx)
    
    # Append first and last element to do proper interpolation
    t_period = np.append(t_period, t[-1])
    t_period = np.insert(t_period, 0, 0.0)
    
    y_period = np.append(y_period, y_period[-1])
    y_period = np.insert(y_period, 0, y_period[0])
    
    # Get frequency out of period
    y_f = 1.0/y_period
    
    ff = interp1d(t_period, y_f, kind='linear')
    
    f = ff(t)
    
    condition_3 = (A<args.A_min/2.0)|(f<0)
    idx = np.where(condition_3)
    f[idx] = 0.0
    
    return f
    

    

def all_results(net, args):
    
    # Create A and f vectors
    dt           =  args.dt
    exp_len      =  args.exp_len
    warm_up_len  =  args.warm_up_len
    A_max        =  args.A_max
    A_min        =  args.A_min
    T_max        =  args.T_max
    T_min        =  args.T_min
    phi_min      =  args.phi_min
    phi_max      =  args.phi_max
    
    exp_len_test = exp_len*3
    
    device = get_device()
    
    savepath = './save/'

    N_A = int(A_max/A_min)
    N_f = int(T_max/T_min)
    
    A = (np.arange(2,N_A+1)-0.5)*A_min
    f = (np.arange(2,N_f+1)-0.5)*(1.0/T_max)
    
    # Set net to eval state
    net.eval()
    
    waves = np.zeros((N_A-1, N_f-1, exp_len_test))
    Phis = np.zeros( (N_A-1, N_f-1) )
    
    print('Predicting waves!')
    for i in tqdm(range(N_A-1)):
        for j in range(N_f-1):
            
            # Create data vector
            Phi = np.random.uniform(low = 2*phi_min, high = 2*phi_max)
            t = np.arange(0, exp_len_test)*dt
            
            y = A[i]*np.sin(t*f[j]*2*np.pi+Phi)
            
            features = torch.from_numpy(y[:-1]).float().unsqueeze(0).to(device)
            
            
            net.reset()
            net.initialize_sequence(features)
            output = net(exp_len_test-warm_up_len, terminate = True)
            output = output.squeeze().detach().cpu().numpy()
            waves[i,j,:] = output
            Phis[i,j] = Phi
            #print('Done with '+ str(i)+','+str(j))
        
    
    A_evolutions = np.zeros((N_A-1, N_f-1, exp_len_test))
    f_evolutions = np.zeros((N_A-1, N_f-1, exp_len_test))
    print('Calculating amplitudes and frequencies!')
    for i in tqdm(range(N_A-1)):
        for j in range(N_f-1):
            A_evolutions[i,j,:] = Calculate_A(waves[i,j,:], args)
            f_evolutions[i,j,:] = Calculate_f(waves[i,j,:], args, A_evolutions[i,j,:])
    
        
    np.save( savepath+'waves.npy',         waves)
    np.save( savepath+'phi.npy',           Phis)
    np.save( savepath+'A.npy',             A_evolutions)
    np.save( savepath+'f.npy',             f_evolutions)
    print('Shape of "waves" is '+str(waves.shape))
    
def err_f(evolutions, target):
    return (evolutions-target)/target

def plot_single (A_user, f_user, args):
    
    A_max        =  args.A_max
    A_min        =  args.A_min
    T_max        =  args.T_max
    T_min        =  args.T_min
    
    warm_up_len  =  args.warm_up_len
    
    savepath = './save/'
    
    N_A = int(A_max/A_min)
    N_f = int(T_max/T_min)
    
    A_array = np.arange(1,N_A+1)*A_min
    f_array = (np.arange(1,N_f+1)-0.5)*(1.0/T_max)
    
    A, A_idx =  find_nearest(A_array, A_user)
    f, f_idx =  find_nearest(f_array, f_user)
    
    Phi = np.load(savepath+'phi.npy') 
    Phi = Phi[A_idx, f_idx]
    
    
    
    wave = np.load(savepath+'waves.npy')[A_idx, f_idx, :] 
    A_evolution = np.load(savepath+'A.npy')[A_idx, f_idx, :]
    f_evolution = np.load(savepath+'f.npy')[A_idx, f_idx, :]
    
    t = np.arange(0,wave.shape[0])*args.dt
    
    wave_target = A*np.sin(t*f*2*np.pi+Phi)
    
    A_err = err_f(A_evolution, A)
    f_err = err_f(f_evolution, f)
    
    
    fig, axs = plt.subplots(3, 1, figsize=(30, 14), sharex=True) # share x axis so zoom zooms all plots
    plt.sca(axs[0])
    plt.title('Predict future values for time sequences \n dt = 1ms', fontsize=30)
    plt.xlabel('t [ms]', fontsize=20)
    plt.ylabel('Amplitude [-]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(t,wave_target, 'b', label = 'Target values')
    plt.plot(t[:warm_up_len],wave[:warm_up_len], 'r', label = 'Network warm-up')
    plt.plot(t[warm_up_len:],wave[warm_up_len:], 'r:', label = 'Predicting next points iteratively')
    plt.legend(fontsize=20, loc = 'right')
    
    
    plt.sca(axs[1])
    plt.plot(t, A_evolution, 'r',  label = 'Amplitude')
    plt.plot(t, np.zeros_like(t)+A, "--", color="black", label = 'Target Amplitude')
    plt.ylabel('Amplitude [-]', fontsize=20)
    plt.xlabel('t [ms]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax2 = axs[1].twinx()
    plt.sca(ax2)
    plt.plot(t, f_evolution, 'g',  label = 'Frequency')
    plt.plot(t, np.zeros_like(t)+f, "--", color="gray", label = 'Target Frequency')
    plt.ylabel('frequency [Hz]', fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(fontsize=20, loc = 'right')
    
    handles, labels = [(a + b) for a, b in zip(axs[1].get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    plt.legend(handles, labels, fontsize=20, loc = 'right')
    
    
    plt.sca(axs[2])
    plt.plot(t, A_err, 'r',  label = 'Amplitude Error (prediction-target)/target')
    plt.ylabel('Amplitude [-]', fontsize=20)
    plt.xlabel('t [ms]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax3 = axs[2].twinx()
    plt.sca(ax3)
    plt.plot(t, f_err, 'g',  label = 'Frequency Error (prediction-target)/target')
    plt.ylabel('frequency [-]', fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(fontsize=20, loc = 'right')
    
    handles, labels = [(a + b) for a, b in zip(axs[2].get_legend_handles_labels(), ax3.get_legend_handles_labels())]
    plt.legend(handles, labels, fontsize=20, loc = 'right')
    
    
def plot_all(args):
    
    A_max        =  args.A_max
    A_min        =  args.A_min
    T_max        =  args.T_max
    T_min        =  args.T_min
    
    warm_up_len  =  args.warm_up_len
    
    
    N_A = int(A_max/A_min)
    N_f = int(T_max/T_min)
    
    A = np.arange(1,N_A+1)*A_min
    f = (np.arange(1,N_f+1)-0.5)*(1.0/T_max)
    
    
    A_evolution = np.load(args.savepath+'A.npy')
    f_evolution = np.load(args.savepath+'f.npy')
    
    t = np.arange(0,A_evolution.shape[2])*args.dt
    
    
    
    # https://stackoverflow.com/questions/54659559/matplotlib-python-connect-two-scatter-plots-with-lines-for-each-pair-of-x-y-va
    
    ########################################################################
    # Slider
    ########################################################################

    # Draw Plots, from top position, angle, motor
    fig, axs = plt.subplots(1, 1,figsize=(14, 10), sharex=True)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.3, top=0.9)
    
    # plots = [[0] * N_A for i in range(N_f)]
    
    
    A_array = np.array([A,]*f.shape[0])
    f_array = np.array([f,]*A.shape[0]).transpose()
    scatter_ref     =        plt.scatter(A_array, f_array, color='y')
    idx_start = 500
    scatter_pred         =        scatter(A_evolution[:,:,idx_start],f_evolution[:,:,idx_start], color='g')
    
    # for i,j in zip(range(N_A), range(N_f)):
    #     plt.plot([A[i],A_evolution[i,j,0]], [f[j],f_evolution[i,j,0]])
    #     #plots[i,j] = plt.plot([A[i],A_evolution[i,j,0]], [f[j],f_evolution[i,j,0]])

# 
    axs.set_xlabel('Amplitude (-)', fontsize=20)
    axs.set_ylabel("Frequency (Hz)", fontsize=20)
    plt.ylim(0, 150)
    plt.xlim(0, 2.0)
    #axs.legend(fontsize=12)

    
    axcolor = 'lightgoldenrodyellow'
    axtstep = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)

    # Sliders
    t0 = 0.0
    delta_step = args.dt
    ststep = Slider(axtstep, 'Timestep', valmin = 0.0, valmax = max(t), valinit=0.0,
                    valstep=delta_step)


    def update(val):
        plt.sca(axs)
        t_curr             =      float(ststep.val)  # Put the initial timestep at the first timestep after the first possible context window
        t_curr, t_idx      =      find_nearest(t, t_curr)
        print(t_idx)
        # xx = np.vstack ((A_evolution[:,:,t_idx], f_evolution[:,:,t_idx]))
        # scatter_pred.set_offsets (xx.T)
        plt.cla()
        scatter_ref          =        plt.scatter(A_array, f_array, color='y')
        scatter_pred         =        scatter(A_evolution[:,:,t_idx],f_evolution[:,:,t_idx], color='g')
        axs.set_xlabel('Amplitude (-)', fontsize=20)
        axs.set_ylabel("Frequency (Hz)", fontsize=20)
        plt.ylim(0, 150)
        plt.xlim(0, 2.0)
        # for i,j in zip(range(N_A), range(N_f)):
        #     plots[i,j] = plt.plot([A[i],A_evolution[i,j,t_idx]], [f[j],f_evolution[i,j,t_idx]])
        plt.draw()
        #plt.sca(axtstep)

    ststep.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        ststep.reset()


    button.on_clicked(reset)

    plt.show()



def plot_single_raw(net, A, f, args):
    
    device = get_device()

    
    phi_min      =  args.phi_min
    phi_max      =  args.phi_max
    
    exp_len      =  args.exp_len
    warm_up_len  =  args.warm_up_len
    

    

    
    Phi = np.random.uniform(low = phi_min, high = phi_max)
    t = np.arange(0,args.exp_len)*args.dt
    
    wave_target = A*np.sin(t*f*2*np.pi+Phi)
    
        
    # Set net to eval state
    net.eval()
    
    features = torch.from_numpy(wave_target[:-1]).float().unsqueeze(0).to(device)

    
    net.reset()
    net.initialize_sequence(features)
    output = net(exp_len-args.warm_up_len, terminate = True)
    output = output.squeeze().detach().cpu().numpy()
    net.reset()
    
    wave = output
    A_evolution = Calculate_A(output, args)
    f_evolution = Calculate_f(output, args, A_evolution)
    
       
    
    fig, axs = plt.subplots(2, 1, figsize=(30, 14), sharex=True) # share x axis so zoom zooms all plots
    plt.sca(axs[0])
    title_string  = 'Predict future values for time sequences \n dt = 1ms \n A = '+str(A)+', f = '+str(f)+'Hz'
    plt.title(title_string, fontsize=30)
    plt.xlabel('t [ms]', fontsize=20)
    plt.ylabel('Amplitude [-]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(t,wave_target, 'b', label = 'Target values')
    plt.plot(t[1:warm_up_len+1],wave[:warm_up_len], 'r', label = 'Network warm-up')
    plt.plot(t[warm_up_len+1:],wave[warm_up_len:-1], 'r:', label = 'Predicting next points iteratively')
    plt.legend(fontsize=20, loc = 'right')
    
    
    plt.sca(axs[1])
    plt.plot(t[warm_up_len:], A_evolution[warm_up_len:], 'r',  label = 'Amplitude')
    plt.plot(t, np.zeros_like(t)+A, "--", color="black", label = 'Target Amplitude')
    plt.ylabel('Amplitude [-]', fontsize=20)
    plt.xlabel('t [ms]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax2 = axs[1].twinx()
    plt.sca(ax2)
    if f_evolution is not None:
        plt.plot(t[warm_up_len:], f_evolution[warm_up_len:], 'g',  label = 'Frequency')
    plt.plot(t, np.zeros_like(t)+f, "--", color="gray", label = 'Target Frequency')
    plt.ylabel('frequency [Hz]', fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(fontsize=20, loc = 'right')
    
    handles, labels = [(a + b) for a, b in zip(axs[1].get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    plt.legend(handles, labels, fontsize=20, loc = 'right')
    
    
    plt.show()
    plt.pause(0.001)



def plots(net, args):
    try:
        plot_single_raw(net, 0.5, 100.0, args)
        plt.pause(0.01)  
    except:
        print('Plot 1 could not be done')
        
    # try:
    #     plot_single_raw(net, 0.8, 50.0, args)
    #     plt.pause(0.01)  
    # except:
    #     print('Plot 2 could not be done')
        
    # try:
    #     plot_single_raw(net, 0.5, 200.0, args)
    #     plt.pause(0.01)  
    # except:
    #     print('Plot 3 could not be done')
        
    # try:
    #     plot_single_raw(net, 0.2, 400.0, args)
    #     plt.pause(0.01)  
    # except:
    #     print('Plot 4 could not be done')
    time.sleep(1)
    