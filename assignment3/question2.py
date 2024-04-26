import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.restoration import estimate_sigma
from skimage.restoration import denoise_wavelet

# Function to add white Gaussian noise to an image
def add_gaussian_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

# Function to compute low pass Gaussian filter denoising
def gaussian_filter_denoise(image, filter_length, sigma):
    denoised = cv2.GaussianBlur(image, (filter_length, filter_length), sigma)
    return denoised

# Function to compute adaptive MMSE denoising
def AdaptiveMSME(orig, noisy):

    noisy_low_pass = cv2.GaussianBlur(noisy, (3,3), 1)
    noisy_high_pass = noisy - noisy_low_pass
    kernel = cv2.getGaussianKernel(3, 1)
    kernel=kernel*kernel.T
    kernel[1,1]-=1
    mul=np.sum(kernel**2)
    varz = mul*100

    stride=16

    resultimg=np.zeros(noisy.shape)
    count_img = np.zeros(noisy.shape)

    for x in range(0,noisy_high_pass.shape[0],stride):
        for y in range(0,noisy_high_pass.shape[1],stride):
            xend=min(x+32,noisy_high_pass.shape[0])
            yend=min(y+32,noisy_high_pass.shape[1])
            patch=noisy_high_pass[x:xend,y:yend]
            varx = np.var(patch)-varz
            resultimg[x:xend,y:yend]+=patch*varx/(varx+varz)
            count_img[x:xend, y:yend] += 1
    
    
    copy_resultimg = resultimg.copy()
    
    resultimg/=count_img
    

    # copy_resultimg[16:noisy.shape[0]-16,16:noisy.shape[1]-16]/=3
    # print(count_img[16:noisy.shape[0]-16,16:noisy.shape[1]-16])
    # copy_resultimg[:16,16:noisy.shape[1]-16]/=2
    # copy_resultimg[noisy.shape[0]-16:,16:noisy.shape[1]-16]/=2
    # copy_resultimg[16:noisy.shape[0]-16,:16]/=2
    # copy_resultimg[16:noisy.shape[0]-16,noisy.shape[1]-16:]/=2
    
    
    
    # resultimg[16:img.shape[0]-16,16:img.shape[1]-16]/=3
    # resultimg[:16,16:img.shape[1]-16]/=2
    # resultimg[img.shape[0]-16:,16:img.shape[1]-16]/=2
    # resultimg[16:img.shape[0]-16,:16]/=2
    # resultimg[16:img.shape[0]-16,img.shape[1]-16:]/=2
    
    
    resultimage=resultimg+noisy_low_pass
    # print(np.sum(np.abs(resultimg-copy_resultimg)))
    # assert np.equal(resultimg,copy_resultimg).all()
    
    error = np.sum((resultimage-orig)**2)/(orig.shape[0]*orig.shape[1])
    print(f"the MSE error for adaptive MMSE is {error}")
    plt.imshow(resultimage,cmap="gray")
    plt.axis('off')
    plt.savefig('adaptive_mmse.png')

def sure(img,vz):
    best=0
    minerr=float('inf')
    tlist=np.linspace(0,50,1000)
    for t in tlist:
        estm=np.sum((np.minimum(img,t))**2)-2*vz*np.sum(np.abs(img)<t)
        if estm<minerr:
            minerr=estm
            best=t
    return best


def AdaptiveShrinkage(orig,noisy):
    noisy_low_pass = cv2.GaussianBlur(noisy, (3,3),0)
    noisy_high_pass = noisy - noisy_low_pass
    high_pass_result=np.zeros(orig.shape)
    stride=32
    kernel = cv2.getGaussianKernel(3, 1)
    kernel=kernel*kernel.T
    kernel[1,1]-=1
    mul=np.sum(kernel**2)
    varz = mul*100
    for x in range(0,noisy.shape[0],stride):
        for y in range(0,noisy.shape[1],stride):
            xend=min(x+32,noisy_high_pass.shape[0])
            yend=min(y+32,noisy_high_pass.shape[1])
            patch=noisy_high_pass[x:xend,y:yend]
            t=sure(patch,varz)
            high_pass_result[x:xend,y:yend]= np.sign(patch)*np.maximum((np.abs(patch) - t), 0)

    result=high_pass_result+noisy_low_pass
    err=np.sum((result-orig)**2)/(orig.shape[0]*orig.shape[1])
    print(f"the MSE error for adaptive shrinkage is {err}")
    plt.imshow(result,cmap="gray")
    plt.axis('off')
    plt.savefig('denoised_adaptive_shrinkage.png')

# Load the lighthouse image
image = cv2.imread("lighthouse2.bmp", cv2.IMREAD_GRAYSCALE)

# Convert image to float32
image = image.astype(np.float32)

# Add white Gaussian noise
sigma_Z = 10  # Adjust sigma_Z^2 according to the problem statement
noisy_image = add_gaussian_noise(image, sigma_Z)


# Denoise using low pass Gaussian filter and find best parameters
filter_lengths = [3,7,11]
sigmas = [0.1, 1, 2, 4, 8]

best_mse = float('inf')
best_filter_length = None
best_sigma = None
for length in filter_lengths:
    for sigma in sigmas:
        denoised_image = gaussian_filter_denoise(noisy_image, length, sigma)
        mse = mean_squared_error(image, denoised_image)
        if mse < best_mse:
            best_mse = mse
            best_filter_length = length
            best_sigma = sigma

print(f"Best parameters for Gaussian filter denoising: Filter length={best_filter_length}, Sigma={best_sigma}, MSE={best_mse}")

# Denoise using adaptive MMSE
denoised_mmse = AdaptiveMSME(image,noisy_image)
# mmse_mse = mean_squared_error(image, denoised_mmse)
# print(f"MSE for adaptive MMSE denoising: {mmse_mse}")

# Denoise using adaptive shrinkage
denoised_shrinkage = AdaptiveShrinkage(image, noisy_image)
# shrinkage_mse = mean_squared_error(image, denoised_shrinkage)
# print(f"MSE for adaptive shrinkage denoising: {shrinkage_mse}")
