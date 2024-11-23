import cv2
import matplotlib.pyplot as plt
import numpy as np

def display_images(l, titles=None, fontsize=12):
    n=len(l)
    fig,ax = plt.subplots(1,n)
    for i,im in enumerate(l):
        ax[i].imshow(im)
        ax[i].axis('off')
        if titles is not None:
            ax[i].set_title(titles[i], fontsize=fontsize)
    fig.set_size_inches(fig.get_size_inches()*n)
    plt.tight_layout()
    plt.show()
    
#Loading Images
im = cv2.imread(r'c:\Users\phenr\Pictures\Screenshots\Screenshot 2024-11-23 205326.png')
print(im.shape)
plt.imshow(im)
plt.show()

bw_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(bw_im.shape)
plt.imshow(bw_im)
plt.show()
