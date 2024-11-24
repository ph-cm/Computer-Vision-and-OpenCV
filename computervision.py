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

#Braille Image Processing
im = cv2.blur(bw_im, (3, 3))
im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 4)
im = cv2.medianBlur(im,3)
_,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
im = cv2.GaussianBlur(im, (3, 3), 0)
_,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
plt.imshow(im)
plt.show()

orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=8)
f,d = orb.detectAndCompute(im,None)
print(f"First 5 points: { [f[i].pt for i in range(5)] }")

#Testing to fix the plot_dot function
#im_with_keypoints = cv2.drawKeypoints(im, f, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(im_with_keypoints, cmap="gray")
#plt.title("Pontos Detectados na Imagem Original")
#plt.show()
sift = cv2.SIFT_create()
f, d = sift.detectAndCompute(im, None)

# Exibir os pontos detectados
im_with_keypoints = cv2.drawKeypoints(im, f, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(im_with_keypoints, cmap="gray")
plt.title("Dots Detected using SIFT")
plt.show()

def plot_dots(dots):
    img = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    for x in dots:
        cv2.circle(img, (int(x[0]), int(x[1])), 3, 255, -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
pts = [x.pt for x in f]
plot_dots(pts)

min_x, min_y, max_x, max_y = [int(f([z[i] for z in pts])) for f in (min,max) for i in (0,1)]
min_y+=13
plt.imshow(im[min_y:max_y, min_x:max_x], cmap="gray")
plt.show()

off = 5
src_pts = np.array([(min_x-off, min_y-off), (min_x-off, max_y+off), (max_x+off,min_y-off),(max_x+off,max_y+off)])
w = int(max_x-min_x+off*2)
h = int(max_y-min_y+off*2)
dst_pts = np.array([(0,0),(0,h),(w,0),(w,h)])
ho,m = cv2.findHomography(src_pts, dst_pts)
trim = cv2.warpPerspective(im,ho,(w,h))
plt.imshow(trim)
plt.show()

char_h = 36
char_w = 24
def slice(img):
    dy, dx = img.shape
    y = 0
    while y+char_h<dy:
        x = 0
        while x+char_w<dx:
            if np.max(img[y:y+char_h,x:x+char_w])>0:
                yield img[y:y+char_h, x:x+char_w]
            x+=char_w
        y+=char_h
sliced = list(slice(trim))
display_images(sliced)
