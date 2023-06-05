import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
#matplotlib inline
from scipy import signal
from scipy import ndimage

def get_data_path(run):
    root_dir = '/Users/clairewu/Documents/F22/BTS-ARMY'
    return f'{root_dir}/S{run}'

def get_image_data(run, scan_num):
    data_path = get_data_path(run)
    image = Image.open(f'{data_path}/BTS_test_1_S{run}_{str(scan_num).zfill(5)}.tif')  # import .tiff files from folder MapSacan. run is the run number and i is the number of scans
    return np.array(image)   
# # Generating toy image

# N1 = 400
# N2 = 1000
# # generate a random background image
# X = 1 * np.random.randn(N1,N2)

# # generate random coords for 2 centroids
# L = 100
# x1 = np.random.randint(L,N2-L)
# y1 = np.random.randint(L,N1-L)
# print(x1, y1)

# x2 = np.random.randint(L,N2-L)
# y2 = np.random.randint(L,N1-L)
# print(x2, y2)

# # place centroids on image

# # random intensity calues
# A1 = np.random.uniform(100,255)
# A2 = np.random.uniform(100,255)

# r = 15
# X = cv2.circle(X,(x1,y1),r,color=A1,thickness=-1)
# X = cv2.circle(X,(x2,y2),r,color=A2,thickness=-1)

# plt.imshow(X)
# plt.colorbar()

# # normalize image
# # places intensity values within [0,1]
# X = X - np.min(X,axis=(0,1))
# X = X / np.max(X,axis=(0,1)) 
# X = X * 255 # grayscaled
# plt.hist(X,bins=5)

# thresh = 0.7 # a 'guess' for where to threshold
run = 625
output_dir = get_data_path(run) + "/output_images"
os.makedirs(output_dir, exist_ok=True)
for scan_num in range(21):
    image_data = get_image_data(run, scan_num)
    image_data = np.float32(image_data)
    image_data = (image_data - np.min(image_data,axis=(0,1))) / np.max(image_data,axis=(0,1)) 
    image_data = image_data * 255 # grayscaled
    #plt.hist(image_data,bins=5) #check of intensity spread
    thresh = 0.5 * 255
    _,X_bw = cv2.threshold(image_data,thresh,1,cv2.THRESH_BINARY)
    X_bw.astype(np.uint8)
    plt.imshow(X_bw)
    plt.savefig(f'{output_dir}/BTS_test_1_S{run}_{str(scan_num).zfill(5)}.png', bbox_inches='tight')

thresh = 0.5 * 255
_,X_bw = cv2.threshold(image_data,thresh,1,cv2.THRESH_BINARY)
# _,X_bw = cv2.threshold(X,0,255,cv2.THRESH_BINARY_INV,cv2.THRESH_OTSU)
X_bw.astype(np.uint8)
# plt.imshow(X_bw)
# plt.colorbar()
# plt.set_cmap('gray')


# image matrix input needs to be casted as an integer array
threshold = X_bw.astype(np.uint8)
retval,labels,stats,centroids = cv2.connectedComponentsWithStats(threshold,8,cv2.CV_32S)

print(retval) # number of CC's
print(labels) # each blob is labeled 0 = background, 1,2,3... are the objects
print(stats) 
print(centroids)

print(np.unique(labels))