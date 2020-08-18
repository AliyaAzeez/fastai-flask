from flask import Flask, escape, request,render_template
from keras.models import Sequential, load_model
from fastai.basics import *
from PIL import Image
from torchvision import *
from fastai.vision import *
import torch
import torchvision.transforms as transforms
import glob
import numpy as np
import time 
from torch.utils.data import DataLoader, Dataset
import PIL
import boto3
from datetime import datetime
import pytz
import cv2
from io import BytesIO

def find_appropriate_lr(model:Learner, lr_diff:int = 5, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    assert(lr_diff < len(losses))
    loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #Search for index in gradients where loss is lowest before the loss spike
    #Initialize right and left idx using the lr_diff as a spacing unit
    #Set the local min lr as -1 to signify if threshold is too low
    local_min_lr = -1
    r_idx = -1
    l_idx = r_idx - lr_diff
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > loss_threshold):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1

    lr_to_use = local_min_lr * adjust_value
    
    # if plot:
    #     ## Default set to False
    #     # plots the gradients of the losses in respect to the learning rate change
    #     plt.plot(loss_grad)
    #     plt.plot(len(losses)+l_idx, loss_grad[l_idx],markersize=10,marker='o',color='red')
    #     plt.ylabel("Loss")
    #     plt.xlabel("Index of LRs")
    #     plt.show()

    #     plt.plot(np.log10(lrs), losses)
    #     plt.ylabel("Loss")
    #     plt.xlabel("Log 10 Transform of Learning Rate")
    #     loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
    #     plt.plot(np.log10(lr_to_use), loss_coord, markersize=10,marker='o',color='red')
    #     plt.show()
        
    return lr_to_use


def plot_save_confusion_matrix(self,cm_path, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", slice_size:int=1,
                              norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None, **kwargs)->Optional[plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        '''
         Arguments:
            1.Classification report of learner
            2.cm_path: path where confusion matrix
        
        '''
        # This function is mainly copied from the sklearn docs
        cm = self.confusion_matrix(slice_size=slice_size)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(self.data.c)
        plt.xticks(tick_marks, self.data.y.classes, rotation=90)
        plt.yticks(tick_marks, self.data.y.classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(self.data.y.classes)-.5,-.5)
                           
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        plt.savefig(cm_path)
        if ifnone(return_fig, defaults.return_fig): return fig