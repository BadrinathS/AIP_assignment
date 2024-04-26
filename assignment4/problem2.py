from math import log10, sqrt 
import cv2 
from skimage.metrics import structural_similarity as ssim

import numpy as np 
import lpips
import os
import torch
import scipy.io as sio



def spearman_rank_correlation(x, y):
    # Convert lists to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Rank the data
    rank_x = np.argsort(x).argsort()
    rank_y = np.argsort(y).argsort()

    # Calculate differences between ranks
    diff = rank_x - rank_y

    # Calculate the Spearman correlation coefficient
    n = len(x)
    rho = 1 - (6 * np.sum(diff**2)) / (n * (n**2 - 1))

    return rho


  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def SSIM_score(original, compressed):
    score = ssim(original, compressed, channel_axis=2)
    # diff = (diff * 255).astype("uint8")
    # print(score)
    return score

loss_fn_vgg = lpips.LPIPS(net='vgg')
def LPIPS(original, compressed):
    
    inp, gt = torch.from_numpy(original).permute(2,1,0).unsqueeze(0), torch.from_numpy(compressed).permute(2,1,0).unsqueeze(0)

    inp = 2*((inp - inp.min())/(inp.max()-inp.min())) -1
    gt = 2*((gt - gt.min())/(gt.max()-gt.min())) -1
    # print(inp.min(), inp.max(), gt.min(), gt.max())
    
    with torch.no_grad():
        score = loss_fn_vgg(inp, gt)
    # print(score.item())

    return score.item()



dataset_path = '/mnt/external_ssd/badrinath/AIP_assignment4/hw5'

distorted_path = os.path.join(dataset_path, 'gblur')
refimage_path = os.path.join(dataset_path, 'refimgs')


mat_file = sio.loadmat(os.path.join(dataset_path, 'hw5.mat'))

infotext_path = '/mnt/external_ssd/badrinath/AIP_assignment4/hw5/gblur/info.txt'
info_file = open(infotext_path, "r+")
# print(len(info_file.readlines()))
# exit()

refnames_blur = mat_file['refnames_blur']
blur_dmos = mat_file['blur_dmos']
blur_orgs = mat_file['blur_orgs']

# print(blur_dmos.shape)
# exit()


ref2img = {}
ref2img_indicator = {}

for i, name in enumerate(refnames_blur[0]):
    if name[0] in ref2img:
        ref2img[name[0]].append('img{}.bmp'.format(i+1))
    else:
        ref2img[name[0]] = ['img{}.bmp'.format(i+1)]
        # print('img{}.bmp'.format(i))

ref2img_scores = {}
psnr_accumulate, ssim_accumulate, lpips_accumulate = [], [], []
gt_dmos = []

for key, val in ref2img.items():
    # print("here")
    ref_img = cv2.imread(os.path.join(refimage_path, key), cv2.IMREAD_COLOR)
    for blurname in val:
        
        blur_img = cv2.imread(os.path.join(distorted_path, blurname), cv2.IMREAD_COLOR)
        # print(blurname)
        # print(blur_img.shape, ref_img.shape)
        psnr_sc, ssim_sc, lpips_sc = PSNR(ref_img, blur_img), SSIM_score(ref_img, blur_img), LPIPS(ref_img, blur_img)
        # print(psnr_sc, ssim_sc, lpips_sc)

        if key in ref2img_scores:
            ref2img_scores[key].append((blurname, psnr_sc, ssim_sc, lpips_sc))
        else:
            ref2img_scores[key] = [(blurname, psnr_sc, ssim_sc, lpips_sc)]

        index = int(blurname.split('.bmp')[0].split('img')[-1])
        # print(index, blurname)
        if blur_orgs[0][index-1] == 0:
            psnr_accumulate.append(psnr_sc)
            ssim_accumulate.append(ssim_sc)
            lpips_accumulate.append(lpips_sc)
            gt_dmos.append(blur_dmos[0][index-1])




spearman_psnr, spearman_ssim, spearman_lpips = spearman_rank_correlation(psnr_accumulate, gt_dmos), spearman_rank_correlation(ssim_accumulate, gt_dmos), spearman_rank_correlation(lpips_accumulate, gt_dmos)

print("Correlation, PSNR {}, SSIM {}, LPIPS {}".format(spearman_psnr, spearman_ssim, spearman_lpips))