# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 07:59:29 2020

@author: Marcin
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np


import ParseArgs





class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, color='A'):
        
        self.args = ParseArgs.args()
        print(self.args.__dict__)
        
    
        
        self.A_max        =  self.args.A_max
        self.A_min        =  self.args.A_min
        self.T_max        =  self.args.T_max
        self.T_min        =  self.args.T_min
        self.savepath = './save/'
        self.warm_up_len  =  self.args.warm_up_len
        
        
        self.N_A = int(self.A_max/self.A_min)
        self.N_f = int(self.T_max/self.T_min)
        
        self.A = (np.arange(2,self.N_A+1)-0.5)*self.A_min
        self.f = (np.arange(2,self.N_f+1)-0.5)*(1.0/self.T_max)
        
        
        self.A_evolution = np.load(self.savepath+'A.npy')
        self.f_evolution = np.load(self.savepath+'f.npy')
        
        self.t = np.arange(0,self.A_evolution.shape[2])*self.args.dt
        
        self.A_array = np.array([self.A,]*self.f.shape[0]).transpose()
        self.f_array = np.array([self.f,]*self.A.shape[0])
        


        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(2, 1,figsize=(14, 10))
        self.Slider = Rectangle((0.0,0.0), 0.5, 1.0)
        

        if color == 'A':
            self.colours_ref = self.A_array.flatten()
        elif color == 'f':
            self.colours_ref = self.f_array.flatten()
            
        scatter_ref = self.ax[0].scatter( 
                                         self.f_array.flatten(),
                                         self.A_array.flatten(),
                                         vmin = min(self.colours_ref),
                                         vmax = max(self.colours_ref),
                                         c = self.colours_ref,
                                         cmap='magma',
                                         alpha = 0.3,
                                         edgecolor='k')
        
        self.ax[0].set_ylabel('Amplitude (-)', fontsize=20)
        self.ax[0].set_xlabel("Frequency (Hz)", fontsize=20)
        self.ax[0].axis([0.0, 125, 0.0, 1.5])
        # scatter_ref.set_array(self.colours_ref)
        
        # Set y limits
        self.ax[1].add_patch(self.Slider)
        self.ax[1].set(xlim = (0.0,max(self.t)))
        
        self.ax[1].set_xlabel('Time (s)', fontsize=20)
        # Apply scaling
        self.ax[1].set_aspect("equal")
        
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig,
                                           self.update,
                                           frames = 3*self.args.exp_len,
                                           interval=25, 
                                           init_func=self.setup_plot,
                                           blit=True)
        self.ani.save(color+'.mp4', writer=writer)
        plt.show()

    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        y = self.A_evolution[:,:,0].flatten()
        x = self.f_evolution[:,:,0].flatten()
        self.scat = self.ax[0].scatter(x,y,
                                       vmin = min(self.colours_ref),
                                       vmax = max(self.colours_ref),
                                       c = self.colours_ref,
                                       cmap='magma')
        
        ## Lower Chart with Slider

        # Remove ticks on the y-axes
        self.ax[1].yaxis.set_major_locator(plt.NullLocator())
        self.vline = self.ax[1].axvline(self.warm_up_len*self.args.dt, c = 'r', lw = 2)

        
        
        
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.Slider, self.vline,



    def update(self, i):
        """Update the scatter plot."""
        # data = next(self.stream)
        y = self.A_evolution[:,:,i].flatten()
        x = self.f_evolution[:,:,i].flatten()
        
        idx_del = np.where(y<0.01)
        x =  np.delete(x,idx_del)
        y =  np.delete(y,idx_del)
        newcolor = np.delete(self.colours_ref, idx_del)
        
        x = x[:,None]
        y = y[:,None]
        
        

        
        data = np.hstack((x,y))
        # Set x and y data...
        self.scat.set_offsets(data)
        self.scat.set_array(newcolor)
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # # Set colors..
        # self.scat.set_array(data[:, 3])
        # self.scat.set_array(self.A_evolution[:,:,0].flatten())
        
        #Draw SLider
        self.Slider.set_width(i*self.args.dt)
        
        self.vline = self.ax[1].axvline(self.warm_up_len*self.args.dt, c = 'r',  lw = 2)
        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.Slider, self.vline,


if __name__ == '__main__':
    a = AnimatedScatter()
   