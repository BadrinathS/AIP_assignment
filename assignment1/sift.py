from sklearn.decomposition import PCA
from scipy.ndimage import convolve, label
from scipy.ndimage import rotate as rotate_image


import pickle
from sklearn.decomposition import PCA
import os
import numpy as np
import scipy.signal as signal
import cv2
import matplotlib.pyplot as plt


def generate_octaves(img, sigma, s=2):
    k=1.8
  
    imgs=[]
    for _ in range(s+3):
        gImg = cv2.GaussianBlur(img, (5, 5), sigma)
        imgs.append(gImg)
        sigma=sigma*k

    return imgs

def octDogs(imgs):
  list_of_dogs=[]
  for imag1, imag2 in zip(imgs, imgs[1:]):
    list_of_dogs.append(np.subtract(imag1, imag2))

  return list_of_dogs

def generate_dog(scale_space):
    dog_collected=[]
    for imgList in scale_space:
        octList=octDogs(imgList)
        dog_collected.append(octList)
    return dog_collected


def Thresh(img,pos,r=6):
    thresh = ((r+1)**2)/r

    kernel_xx = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
    kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    kernel_yy = kernel_xx.T

    x,y = pos

    box = img[x-1:x+2, y-1:y+2]

    hxx = convolve(box, kernel_xx)
    hxy = convolve(box, kernel_xy)
    hyy = convolve(box, kernel_yy)
    hess = np.array([[hxx[1, 1], hxy[1, 1]], [hxy[1, 1], hyy[1, 1]]])


    # Way to threshold the key point, partly inspired by harris corner style
    
    tr = np.trace(hess)
    dt = np.linalg.det(hess)

    if dt<=0:
        return False
    if ((tr**2)/dt)<thresh:
        return True
    else :
        return False


def find_extrema(dog_images, threshold=0.3, thresh=120):
    keypoints = []
    for j in range(len(dog_images)):
        octImgs=dog_images[j]
        octPts=[]
        for i in range(1,len(octImgs)-1):
            prev,curr,nxt=octImgs[i-1:i+2]
            height, width = curr.shape
            multidimarray = np.stack([prev, curr, nxt],axis=2)
            for x in range(1,height-1):
                for y in range(1,width-1):
                    if curr[x,y]<threshold: # Eliminating weak keypoints
                        continue

                    interestbox = multidimarray[x-1:x+2, y-1:y+2,:]
                    argmax=np.max(interestbox)
                    argmin=np.min(interestbox)
                    
                    if argmax == curr[x,y]:
                        if Thresh(curr,[x,y]): # Keeping strong key points based on gradient strength along axis
                            neighbor = cv2.getRectSubPix(curr, (3, 3), (y, x))
                            contr = np.std(neighbor) #Checking neighbpourhood std is above threshold? 
                            if contr>thresh:
                            
                                octPts.append([[x,y],i])

                    elif argmin==curr[x,y]:
                        if Thresh(curr,[x,y]):
                            neighbor = cv2.getRectSubPix(curr, (3, 3), (y, x))
                            contr = np.std(neighbor)
                            if contr>thresh:
                                octPts.append([[x,y],i+1])
                                
        keypoints.append(octPts)
    return keypoints
                        

def plot_points(image, points, img_name, radius=3, color=(0, 255, 0), thickness=-1):
    dotImg = image.copy()

    for point in points:
        x, y = point
        cv2.circle(dotImg, [y,x], radius, color, thickness)

    plt.imshow(dotImg,cmap='gray')
    plt.axis('off')
    plt.savefig('plot_keypoint_'+img_name + '.png')
    

def get_descriptor(image, x,y):
    
    height, width = image.shape
    
    x_start, x_end = max(0, x-20), min(height, x+21)
    y_start, y_end = max(0, y-20), min(width, y+21)
    
    # Making the dimension of images compatible with graident location
    # in case the key point comes in the corner
    
    padup=paddwn=padleft=padright=0
    if x-20<0:
      padleft=20-x
    if x+20>=height:
      padright=21+x-height
    if y-20<0:
      padup=20-y
    if y+20>=width:
      paddwn=21+y-width

    neighborhood = image[x_start:x_end, y_start:y_end]
    neighborhood = cv2.copyMakeBorder(neighborhood, padleft, padright, padup, paddwn, cv2.BORDER_CONSTANT, value=0)
    
    sobel_x = cv2.Sobel(neighborhood, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(neighborhood, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitudes = np.sqrt(sobel_x**2 + sobel_y**2)
    angles = np.arctan2(sobel_y, sobel_x)
    
    dominant_angle = np.mean(angles)

    adjusted_angles = angles - dominant_angle
    
    gx = magnitudes * np.cos(adjusted_angles * np.pi / 180)
    gy = magnitudes * np.sin(adjusted_angles * np.pi / 180)

    flattened_gx = gx.flatten()
    flattened_gy = gy.flatten()


    combined_gradients = np.concatenate((flattened_gx, flattened_gy))
    
    norm_grad = combined_gradients / np.linalg.norm(combined_gradients)
    # print(neighborhood.shape,(x,y),(x_start,x_end,padleft,padright),(y_start,y_end,padup,paddwn),image.shape)

    return norm_grad


# 

def compute_sift(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512,512))

    num_octaves = 3
    num_intervals = 2
    sigma = 1.6

    scale_space = []

    for _ in range(num_octaves):
        scale_space.append(generate_octaves(img, sigma, num_intervals))
        sigma=sigma*2
        
        img = img[::2,::2]

    dogs = generate_dog(scale_space)

    sift_keypoints = find_extrema(dogs)


    allpoints=[]
    img=scale_space[0][0]
    for i in range(len(scale_space)):
        keys=sift_keypoints[i]
        points = [sublist[0] for sublist in keys]
        allpoints=allpoints+points
        
        
    print("No. of keypoints for " + str(img_path.split('/')[-1][:-4]) + ": ", len(allpoints))
    plot_points(img,allpoints, img_name=img_path.split('/')[-1][:-4])
    
    
    # Process below computes descriptpor for keypoints, 
    # however they are not used in this script for any purposes 
    # so it can be commented to save computing efforts
    
    allDescr=[] 
    for octKeys,octGauss in zip(sift_keypoints,scale_space):
        for i in range(len(octKeys)):
            kpt=octKeys[i]
            x,y=kpt[0]
            img=octGauss[kpt[1]]
            desc=list(get_descriptor(img,x,y))
            allDescr.append(desc)
    

    descriptor_array = np.array(allDescr)
    pca = PCA(n_components=20)
    final_descriptor = pca.fit_transform(descriptor_array)
    
    






compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img1.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img1_scale.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img1_rot.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img1_gauss.png')

compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img2.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img2_rot.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img2_scale.png')
compute_sift(img_path='/data2/badrinath/AIP/assignment1/data/img2_gauss.png')
