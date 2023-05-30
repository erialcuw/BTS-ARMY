#%%
# =============================================================================
# imports
# =============================================================================
import os
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
import cv2
import tifffile
#matplotlib qt

#%%
# =============================================================================
# Below are some functions for reading Tiff file images and calculating Rocking curve,2D map intensity and Centroid.
# =============================================================================
def get_data_path(run,):
    root_dir = '/Users/clairewu/Documents/F22/BTS-ARMY'
    return f'{root_dir}/S{run}'

def get_image_data_np(run, scan_num):
    data_path = get_data_path(run)
    image = Image.open(f'{data_path}/BTS_test_1_S{run}_{str(scan_num).zfill(5)}.tif')  # import .tiff files from folder MapSacan. run is the run number and i is the number of scans
    return np.array(image)                  

def get_image_data(run, scan_num):
    data_path = get_data_path(run)
    image = Image.open(f'{data_path}/BTS_test_1_S{run}_{str(scan_num).zfill(5)}.tif')  # import .tiff files from folder MapSacan. run is the run number and i is the number of scans
    return image   
 
def RockingCurve(run, scan_num,x,y):
    [xmin,xmax]  = x
    [ymin,ymax] = y
    ROI_sum = [0]*(scan_num)   
    for i in range (0,scan_num):
        image_data = get_image_data(run, scan_num)
        ROI_sum[i]=np.sum(image_data[xmin:xmax,ymin:ymax])
    ROI_sum_array = np.asarray(ROI_sum)    
    intensity= ROI_sum_array
    return intensity
 
def Intensity2D(run,x_pixel,y_pixel,x,y):
    scan_num = x_pixel*y_pixel
    [xmin,xmax] = x
    [ymin,ymax] =y
    
    ROI_sum = [0]*(scan_num)
    for i in range (1,scan_num):
        image_data = get_image_data(run, scan_num)
        ROI_sum[i-1]=np.sum(image_data[xmin:xmax,ymin:ymax])
    
        
    ROI_sum_array = np.asarray(ROI_sum)
    intensity= ROI_sum_array.reshape(x_pixel,y_pixel)
    return intensity
 
def Centroid2D(run,x_pixel,y_pixel,x,y):
    scan_num =  x_pixel*y_pixel
    [xmin,xmax] = x
    [ymin,ymax] =y
    
    centroid_x = np.empty(scan_num)
    centroid_y= np.empty(scan_num)
    
    for i in range (1,scan_num):
        image_data = get_image_data(run, scan_num)
        com = ndimage.measurements.center_of_mass(image_data[x_min:x_max,y_min:y_max])
        
        centroid_x[i],centroid_y[i] = com[0],com[1]
    cen_x =  centroid_x.reshape(x_pixel,y_pixel)
    cen_y =  centroid_y.reshape(x_pixel,y_pixel)
    return cen_x,cen_y
 
#%%
# =============================================================================
# Select ROI: In this cell you manually chose the x and y cordinates of the ROI.
# =============================================================================
run = 258
output_dir = get_data_path(run) + "/output_images"
os.makedirs(output_dir, exist_ok=True)
for scan_num in range(31):
    image_data = get_image_data_np(run, scan_num)
    image_data = np.float32(image_data)
    gray = cv2.GaussianBlur(image_data, (5,5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    image_data = cv2.circle(image_data, maxLoc, 25, (255, 0, 0), 2)
    print(maxLoc)
    #cv2.imshow("hello", image_data)
    #cv2.waitKey(0)

    plt.close('all')
    plt.figure(figsize= (10,10))
    c=plt.imshow(image_data, origin='lower', cmap='RdBu',vmin = 0, vmax =10)
    plt.xlim([1,1028])
    plt.ylim([1,512]) #1028 x 512
    cbar=plt.colorbar(c)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/BTS_test_1_S{run}_{str(scan_num).zfill(5)}.png', bbox_inches='tight')
 
#%%
# =============================================================================
# Rocking curve: use the x and y cordinates of ROI from above   
# =============================================================================
run = 625
 
filepath= '/Users/clairewu/Documents/F22/BTS-ARMY'
folderpath = filepath+'/S'%run
cnt = 0
for path in os.listdir(folderpath):
    if os.path.isfile(os.path.join(folderpath,path)):
        cnt+=1
num_of_files = cnt-1  
 
# use the values use selected from the above plot.
#IMPORTANT: x and y coordinates are somehow flipped. What you select as x is actually. I haven't fixed this issue yet.   
x_ROI = [80,380]
y_ROI = [350,445]
 
# call the rocking cruve function. 
RC = RockingCurve(num_of_files,[1, 1028], [1, 512])
th = np.linspace(16.1,16.5, 41) # These values of theta were used in the experiment.
                                # This info. can be found in your log book and if not then the spec file.
 
th_b = 16.29  # Bragg peak angle
del_th = th-th_b # Angular deviation from the Bragg peak. 
 
plt.close('all') 
plt.plot(del_th[9:],RC[9:],marker = 'o', color = 'slateblue',markersize = 7, linewidth=4,alpha = 0.5)
#plt.legend(loc= 'upper right')
plt.xlabel('$\Theta -\Theta_{B} $',fontsize = 15)
plt.ylabel('Intensity',fontsize = 15)
plt.yscale('log')
plt.xlim([-0.1, 0.2])
 
#%%
# =============================================================================
# Intensity: Again use the ROI cordinates by plotting a file of the run you are
# ineterested and use those coordinates here. 
# =============================================================================
run = 57 # Enter the run number. Change the filepath in the Intensity2D function.
x_pixel,y_pixel = 41,41 # Manually put these values from logbook or nc files. 
 
# use the values use selected from the above plot.
#IMPORTANT: x and y coordinates are somehow flipped. What you select as x is actually. I haven't fixed this issue yet.   
x_ROI = [100,400]
y_ROI = [100,400]
 
inten = Intensity2D(run,x_pixel,y_pixel,x_ROI, y_ROI) # this variable stores the intensity values in 2D array form.
 
# NOTE: You can do a normalization of the inensity by chosing a detector image area where the diffraction signal is neglible.
 
# Now you have to give values to the x and y cordinates of the intensity array.
ceny, cenz = -0.87, -1.8 # y and z values of the initial sample stage motors. You can find these values in Logbook, or the nc files.  
rangey,rangez = 0.2,0.2  # range of y and z values scanned. Again, found in logbook or nc file.
 
x_range = np.linspace(ceny-rangey,ceny+rangey,41)
y_range = np.linspace(cenz-rangez,cenz+rangez,41)
 
#plot
plt.close('all')
fig = plt.figure(figsize= (5,5))
ax1=fig.add_subplot(111)
c=ax1.imshow(inten,origin='lower',cmap='viridis',vmax = 1000, extent = [x_range.min(),x_range.max(),y_range.min(),y_range.max()])
ax1.set_xlabel('X (mm)',size = 15)  # NOTE: I have labeled the cordinates as X and Y instead of y and z.
ax1.set_ylabel('Y (mm)',size= 15)
ax1.set_title('Intensity, run{}'.format(run), size = 15)
cbar=plt.colorbar(c,label ='CIntensity')
plt.tight_layout()
 
#%%
# =============================================================================
# Centroid: This section wil calculate the centroid. You can Chose the axis along
# which you want to calculate the centroid. 
# =============================================================================
 
run = 57 # Enter the run number. Change the filepath in the Intensity2D function.
x_pixel,y_pixel = 41,41  # Manually put these values from logbook or nc files. 
# use the values use selected from the above plot.
#IMPORTANT: x and y coordinates are somehow flipped. What you select as x is actually. I haven't fixed this issue yet.   
x_ROI = [100,400]
y_ROI = [100,400]
 
centroid = Centroid2D(run,x_pixel,y_pixel,x_ROI, y_ROI)
centroid_y = centroid[1]
 
#Now you have to give values to the x and y cordinates of the intensity array.
ceny, cenz = -0.87, -1.8 # y and z values of the initial sample stage motors. You can find these values in Logbook, or the nc files.  
rangey,rangez = 0.2,0.2  # range of y and z values scanned. Again, found in logbook or nc file.
 
x_range = np.linspace(ceny-rangey,ceny+rangey,41)
y_range = np.linspace(cenz-rangez,cenz+rangez,41)
 
#plot
plt.close('all')
fig = plt.figure(figsize= (5,5))
ax1=fig.add_subplot(111)
c=ax1.imshow(centroid_y,origin='lower',cmap='viridis',extent = [x_range.min(),x_range.max(),y_range.min(),y_range.max()])
ax1.set_xlabel('X (mm)',size = 15)  # NOTE: I have labeled the cordinates as X and Y instead of y and z.
ax1.set_ylabel('Y (mm)',size= 15)
ax1.set_title('Centroid, run{}'.format(run), size = 15)
#ax1.set_title('Intensity, run{}'.format(run), size = 20)
cbar=plt.colorbar(c,label ='Centroid_Y')
plt.tight_layout()