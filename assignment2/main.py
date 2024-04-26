import cv2
import numpy as np
import math

from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
import pdb

import cv2
import numpy as np
import math

from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pdb

def get_value_from_gaussian(x, sigma):
    return math.exp( -(x**2)/ (sigma**2))

def get_eucledian_distance(a,b):
    return math.sqrt(a**2 + b**2)

class NCut:
    def __init__(self, radius=10, name='default'):
        print('NCUT Algorithm')
        self.sigma_spatial = 10
        self.sigma_intensity = 120
        self.dist_threshold = radius
        self.spatial_lut = self.get_spatial_lut(self.dist_threshold)
        self.intensity_lut = self.get_intensity_lut()
        self.name = name
        
    def get_spatial_lut(self, thresh):
        lut = np.zeros((thresh+1,thresh+1), dtype=np.float64)
        for i in range(thresh+1):
            for j in range(thresh+1):
                lut[i][j] = get_value_from_gaussian(math.sqrt(i*i+j*j), self.sigma_spatial)
        #pdb.set_trace()
        return lut

    def get_intensity_lut(self):
        lut = np.zeros(256, dtype=np.float64)
        for i in range(256):
            lut[i] = get_value_from_gaussian(i, self.sigma_intensity)
        #pdb.set_trace()
        return lut

    def create_graph(self, img):
        # number of nodes : H * W
        N = img.shape[0] * img.shape[1]
        adj_matrix = np.zeros((N, N), dtype=np.float64)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                x_range_min = max(0, x-self.dist_threshold)
                y_range_min = max(0, y-self.dist_threshold)
                x_range_max = min(img.shape[1]-1, x+self.dist_threshold)
                y_range_max = min(img.shape[0]-1, y+self.dist_threshold)    
                for i in range(y_range_min,y_range_max+1): 
                    for j in range(x_range_min,x_range_max+1):
                        val_intensity = self.intensity_lut[abs(img[y,x] - img[i,j])]
                        val_dist = self.spatial_lut[abs(y-i)][abs(x-j)]
                        adj_matrix[y*img.shape[1]+x, i*img.shape[1]+j] = val_intensity * val_dist
                    
        return adj_matrix

    def discretize(self, vec, shape):
        discrete_mask = np.zeros(vec.shape, np.float32)
        continuous_mask = ((vec - vec.min())/(vec.max()-vec.min()))*255
        discrete_mask[vec > 0] = 255
        continuous_mask = continuous_mask.astype(np.uint8)
        continuous_mask = continuous_mask.reshape(shape)
        discrete_mask = discrete_mask.astype(np.uint8)
        discrete_mask = discrete_mask.reshape(shape)
        colored_disc_mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        colored_disc_mask[:,:] = [0,255,0]
        colored_disc_mask[discrete_mask > 0] = [255,0,0]
        #mask[vec > 0] = 255
        return colored_disc_mask, continuous_mask

    def process(self, img):
        img = img.astype(np.int32)
        N = img.shape[0] * img.shape[1]
        W = self.create_graph(img)
        W_img = (W.copy() / W.max()) * 255.0

        W_img = cv2.resize(W_img, (512, 512))

        D = np.zeros((N,N), dtype=np.float64)
        for i in range(N):
            D[i,i] = math.sqrt(W[i,:].sum())
        D_inv = np.linalg.inv(D)
        
        M = np.matmul(D_inv , np.matmul((D - W) , D_inv) )  

        evals, evecs = eigsh(M, k=5, which='LM')
        
        
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(evecs[:,1:])
        segment_img = np.zeros(img.shape)
        segment_img = kmeans.labels_.reshape(segment_img.shape)
        segment_img = 255*((segment_img - segment_img.min())/(segment_img.max() - segment_img.min()))
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         segment_img[i,j] = kmeans.labels_[]
        
        cv2.imwrite('segmented_'+self.name+'.png', segment_img)
        
    

def k_means_segmentation(img, k=3):
    
    pixel_vals = img.reshape((-1,1))
 
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    # k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))
    
    
    return segmented_image            


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, jaccard_score
import json
from torchmetrics import JaccardIndex

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CricketDataset(Dataset):
    def __init__(self, root_dir, image_list, segmentation_map_list, color_to_object_mapping, transform=None):
        self.root_dir = root_dir
        self.image_list = image_list
        self.segmentation_map_list = segmentation_map_list
        self.color_to_object_mapping = color_to_object_mapping
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        segmentation_map_name = self.segmentation_map_list[idx]

        image_path = os.path.join(self.root_dir, 'images', image_name)
        segmentation_map_path = os.path.join(self.root_dir, 'masks', segmentation_map_name)

        image = Image.open(image_path).convert('RGB')
        segmentation_map = np.array(Image.open(segmentation_map_path).convert('RGB').resize((128,128)))

        map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], len(self.color_to_object_mapping)))

        # print(segmentation_map[0,0])
        for i, (keys, val) in enumerate(self.color_to_object_mapping.items()):
            set_val = np.zeros(len(self.color_to_object_mapping))
            set_val[i] = 1.0
            # print(val)
            val_arg = np.nonzero((segmentation_map==val).all(axis=2))#, set_val, np.zeros(len(self.color_to_object_mapping)))
            
            map[val_arg] = set_val
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(segmentation_map, cmap='gray')
            # plt.subplot(1,2,2)
            # plt.imshow(map[:,:,1], cmap='gray')
            # plt.savefig('inside_dataloader.png')
            # print(map[val_arg].max(), map[val_arg].min())

        # map = cv2.resize(map, (128, 128,9))
        
        transform_map = transforms.Compose([
            transforms.ToTensor()
        ])
        map = transform_map(map)
        # map_pil = Image.fromarray(map)
        # print(map_pil.size)
        if self.transform:
            image = self.transform(image)
            # map = self.transform(map_pil)
        
        # print(map[val_arg].max(), map[val_arg].min())
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(segmentation_map, cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(map[1,:,:], cmap='gray')
        # plt.savefig('inside_dataloader.png')
        return image, map



class ResNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNetSegmentation, self).__init__()
        # Load the pretrained ResNet-18 model
        resnet18 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        # Remove the fully connected layers
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        # Add a convolutional layer to replace the last fully connected layer
        self.conv = nn.Conv2d(512, num_classes, kernel_size=1)
        # Add a convolutional layer to replace the second last fully connected layer
        # self.conv_skip = nn.Conv2d(256, num_classes, kernel_size=1)
        # Add an upsampling layer to increase the resolution of the output feature maps
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # Upsample the feature map from the second last layer
        # x_skip = self.conv_skip(x)
        x_skip = self.upsample(x)
        return self.sigmoid(x_skip)



class FCNWithSkip(nn.Module):
    def __init__(self, num_classes):
        super(FCNWithSkip, self).__init__()
        # Load the pretrained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        # Remove the fully connected layers
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        # Add a convolutional layer to replace the last fully connected layer
        self.conv1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv3 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.conv4 = nn.Conv2d(num_classes, num_classes, kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
        self.upsample2 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        # Upsample the feature map from the second last layer
        x_skip1 = self.upsample1(x1)
        x_skip1 = self.conv3(x_skip1)
        x_skip2 = self.upsample2(x2)
        x_skip2 = self.conv4(x_skip2)

        return self.sigmoid(x_skip2 + x_skip1)

# def jaccard_score(output, target, average='macro'):
#     intersection = torch.sum(output & target, dim=(1, 2))
#     union = torch.sum(output | target, dim=(1, 2))
#     jaccard = intersection / union
#     return torch.mean(jaccard) if average == 'macro' else jaccard

if __name__ == '__main__':
   
    # Image 1 

    # img = cv2.imread("./img1.jpeg",cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (80,80))
    # # cv2.imwrite('img1_noiset_input1.jpg', img)
    # # ncut = NCut (radius=img.shape[0]//4, name = 'img1')
    # # ncut.process(img)
    
    # k_means_result = k_means_segmentation(img, k=3)
    # cv2.imwrite('k_means3_img1.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=4)
    # cv2.imwrite('k_means4_img1.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=5)
    # cv2.imwrite('k_means5_img1.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=6)
    # cv2.imwrite('k_means6_img1.png', k_means_result)
    
    
    
    # # Image 2 
    
    # img = cv2.imread("./img2.jpeg",cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (80,80))
    # # cv2.imwrite('img2_noiset_input1.jpg', img)
    # # ncut = NCut (radius=img.shape[0]//4, name = 'img2')
    # # ncut.process(img)
    
    
    # k_means_result = k_means_segmentation(img, k=3)
    # cv2.imwrite('k_means3_img2.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=4)
    # cv2.imwrite('k_means4_img2.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=5)
    # cv2.imwrite('k_means5_img2.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=6)
    # cv2.imwrite('k_means6_img2.png', k_means_result)
    
    
    # # Image 3 
    # img = cv2.imread("./img3.jpeg",cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (80,80))
    # # cv2.imwrite('img3_noiset_input1.jpg', img)
    # # ncut = NCut (radius=img.shape[0]//4, name = 'img3')
    # # ncut.process(img)
    
    
    # k_means_result = k_means_segmentation(img, k=3)
    # cv2.imwrite('k_means3_img3.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=4)
    # cv2.imwrite('k_means4_img3.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=5)
    # cv2.imwrite('k_means5_img3.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=6)
    # cv2.imwrite('k_means6_img3.png', k_means_result)
    
    
    
    # # Image 4
    # img = cv2.imread("./img4.jpeg",cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (80,80))
    # # cv2.imwrite('img4_noiset_input1.jpg', img)
    # # ncut = NCut (radius=img.shape[0]//4, name = 'img4')
    # # ncut.process(img)
    
    
    # k_means_result = k_means_segmentation(img, k=3)
    # cv2.imwrite('k_means3_img4.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=4)
    # cv2.imwrite('k_means4_img4.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=5)
    # cv2.imwrite('k_means5_img4.png', k_means_result)
    # k_means_result = k_means_segmentation(img, k=6)
    # cv2.imwrite('k_means6_img4.png', k_means_result)
    
    
    
    
    
    
    # Load the JSON files
    with open('dataset/train_test_split.json', 'r') as f:
        train_test_split = json.load(f)

    with open('dataset/label2cmap.json', 'r') as f:
        label2cmap = json.load(f)
    
    # model = ResNetSegmentation(len(label2cmap))
    model = FCNWithSkip(len(label2cmap))
    num_epochs = 10
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.functional.binary_cross_entropy_with_logits()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset

    # Define the transformations
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    # ])

    # Create the custom dataset
    train_dataset = CricketDataset(root_dir='dataset', image_list=train_test_split['train'], 
                                segmentation_map_list=train_test_split['train'], 
                                color_to_object_mapping=label2cmap, transform=transform)

    # Create the custom DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = CricketDataset(root_dir='dataset', image_list=train_test_split['test'], 
                                segmentation_map_list=train_test_split['test'], 
                                color_to_object_mapping=label2cmap, transform=transform)

    # Create the custom DataLoader
    val_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # for batch_idx, (image, data) in enumerate(train_loader):
    #     print(image.shape, data.shape, data[0,0,:,:].max(), data[0,0,:,:].min())
    
    # train_dataset = ImageFolder(root='dataset/images', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # val_dataset = ImageFolder(root='dataset/images', transform=transform)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    loss_ep = []
    val_acc = []
    val_iou = []
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            loss_ep.append(loss.item())
            optimizer.step()

    #     # Validation loop
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # _, predicted = torch.max(outputs, 1)
                predicted = torch.where(outputs>=0.5, 1, 0)
                
                # accuracy = accuracy_score(labels.cpu().numpy().reshape(images.shape[0],-1), predicted.cpu().numpy())
                # iou = jaccard_score(labels, predicted, average='macro')

                #Another method
                # labels_2d = labels.permute(0, 2, 3, 1).reshape(-1, len(label2cmap))
                # predicted_2d = predicted.permute(0, 2, 3, 1).reshape(-1)
                
                # # Calculate the IoU for each class separately
                # iou = jaccard_score(labels_2d.cpu().numpy(), predicted_2d.cpu().numpy(), average=None)
                
                accuracy = accuracy_score(labels.cpu().numpy().reshape(-1,1), predicted.cpu().numpy().reshape(-1,1))
                iou = []
                save_img = images[0,:,:,:].cpu().numpy().transpose(1,2,0)
                save_img = 255.0*((save_img-save_img.min())/(save_img.max()-save_img.min()))
                cv2.imwrite('test_image.png', save_img)
                for clas_arg, (keys, values) in enumerate(label2cmap.items()):
                    label_save = 255.0*predicted[0,clas_arg,:,:].cpu().numpy()
                    # label_save = 255.0*((label_save-label_save.min())/(label_save.max()-label_save.min()))
                    cv2.imwrite('test_'+keys+'.png',label_save)
                    iou.append(jaccard_score(labels.cpu().numpy()[:,clas_arg,:,:].reshape(-1,1), predicted.cpu().numpy()[:,clas_arg,:,:].reshape(-1,1), average='macro'))

                mean_iou = np.mean(iou)

                val_acc.append(accuracy)
                val_iou.append(mean_iou)
                print(f'Accuracy: {accuracy}, Mean IOU: {mean_iou}')
    

        plt.figure()
        plt.plot(loss_ep, color='r', label='train loss')
        plt.plot(val_acc, color='b', label='val acc')
        plt.plot(val_iou, color='g', label='val iou')
        plt.legend()
        plt.xlabel('iterations')
        plt.title('Training Stats')
        plt.savefig('training_stats.png')

    
    
    
    
    
    
    
    
    
    