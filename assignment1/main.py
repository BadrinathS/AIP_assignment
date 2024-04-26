import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from scipy.ndimage import convolve, label
from scipy.ndimage import rotate as rotate_image


import pickle
import os
import platform
import numpy as np
import scipy.signal as signal
import cv2
import matplotlib.pyplot as plt


def octaves(shape):
  return 3


def octalImgs(img,sigma,s=2):

  k=1.8
  
  imgs=[]
  for _ in range(s+3):
    gImg = cv2.GaussianBlur(img, (5, 5), sigma)
    imgs.append(gImg)
    sigma=sigma*k

  return imgs

def totalImgs(img,sigma=1.6,s=2):

  n=octaves(img.shape)
  imgList=[]
  for i in range(n):
    octalList=octalImgs(img,sigma,s)
    imgList.append(octalList)
    sigma=sigma*2
    
    img=img[::2,::2]

  return imgList


def octDogs(imgs):
  dogList=[]
  for imag1, imag2 in zip(imgs, imgs[1:]):
    dogList.append(np.subtract(imag1, imag2))

  return dogList

def dogs(listImgList):

  dogAll=[]
  for imgList in listImgList:
    octDogList=octDogs(imgList)
    dogAll.append(octDogList)
  return dogAll


def hessianThresh(img,coo,r=6):
  thresh = ((r+1)**2)/r

  hessian_kernel_xx = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]])
  hessian_kernel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
  hessian_kernel_yy = hessian_kernel_xx.T

  x,y = coo

  box = img[x-1:x+2, y-1:y+2]

  hxx = convolve(box, hessian_kernel_xx)
  hxy = convolve(box, hessian_kernel_xy)
  hyy = convolve(box, hessian_kernel_yy)
  hess = np.array([[hxx[1, 1], hxy[1, 1]], [hxy[1, 1], hyy[1, 1]]])


  tr = hess[0][0]+hess[1][1]
  dt = hess[0][0]*hess[1][1]-hess[0][1]*hess[1][0]

  if dt<=0:
     return False
  if ((tr**2)/dt)<thresh:
    return True
  else :
    return False




def detect_extrema(dog_images, threshold=0.3, contr_thresh=120):
    Allkeypoints = []
    for j in range(len(dog_images)):
      octImgs=dog_images[j]
      octPts=[]
      for i in range(1,len(octImgs)-1):
        prev,curr,nxt=octImgs[i-1:i+2]
        height, width = curr.shape
        imgstk = np.stack([prev, curr, nxt],axis=2)
        for x in range(1,height-1):
          for y in range(1,width-1):
            if curr[x,y]<threshold: # Eliminating weak keypoints
              continue

            box = imgstk[x-1:x+2, y-1:y+2,:]
            boxmax=np.max(box)
            boxmin=np.min(box)

            if boxmax==curr[x,y]:
              if hessianThresh(curr,[x,y]): # Keeping strong key points based on gradient strength along axis
                neighborhood = cv2.getRectSubPix(curr, (3, 3), (y, x))
                contr = np.std(neighborhood) #Checking neighbpourhood std is above threshold? 
                if contr>contr_thresh:
                  
                  octPts.append([[x,y],i])

            elif boxmin==curr[x,y]:
              if hessianThresh(curr,[x,y]):
                neighborhood = cv2.getRectSubPix(curr, (3, 3), (y, x))
                contr = np.std(neighborhood)
                if contr>contr_thresh:
                  #octPts.append([x,y])
                  octPts.append([[x,y],i+1])

      Allkeypoints.append(octPts)
    return Allkeypoints



def draw_dots(image, points, radius=3, color=(0, 255, 0), thickness=-1):
    dotImg = image.copy()

    for point in points:
        x, y = point
        cv2.circle(dotImg, [y,x], radius, color, thickness)

    plt.imshow(dotImg,cmap='gray')
    plt.axis('off')
    plt.show()



def descriptor(image, x, y):
    image_height, image_width = image.shape
    pad_size = 20

    x_start, x_end = max(0, x - 20), min(image_height, x + 21)
    y_start, y_end = max(0, y - 20), min(image_width, y + 21)
    padup=paddwn=padleft=padright=0
    if x-20<0:
      padleft=20-x
    if x+20>=image_height:
      padright=21+x-image_height
    if y-20<0:
      padup=20-y
    if y+20>=image_width:
      paddwn=21+y-image_width

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
    print(neighborhood.shape,(x,y),(x_start,x_end,padleft,padright),(y_start,y_end,padup,paddwn),image.shape)

    return norm_grad



def getDescr(keyPtList,gaussList):
  allDescr=[]
  for octKeys,octGauss in zip(keyPtList,gaussList):
      for i in range(len(octKeys)):
        kpt=octKeys[i]
        x,y=kpt[0]
        img=octGauss[kpt[1]]
        desc=list(descriptor(img,x,y))
        allDescr.append(desc)
  return allDescr




def doPca(descriptors,n=20):
  data = np.array(descriptors)
  pca = PCA(n_components=n)
  result = pca.fit_transform(data)
  return result



uploaded = files.upload()

file_name1= list(uploaded.keys())[0]
image1 = cv2.imread(file_name1)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
inpimg = np.array(image1)

print("NumPy Array Shape:", inpimg.shape)


inp = cv2.cvtColor(inpimg, cv2.COLOR_BGR2GRAY)
gaussList=totalImgs(inp,2)
dogList=dogs(gaussList)

for oct in gaussList:
  for img in oct:
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.show()
for oc in dogList:
  for im in oc:
    plt.imshow(im,cmap='gray')
    plt.axis('off')
    plt.show()
    
sift_keypoints = detect_extrema(dogList)


#ti = cv2.GaussianBlur(i1, (5, 5), 1.6)
ti=inp[::2,::2]
rotated_img = cv2.rotate(ti, rotateCode=cv2.ROTATE_90_CLOCKWISE)
gaussListR=totalImgs(rotated_img,2)
dogListR=dogs(gaussListR)
sift_keypointsR = detect_extrema(dogListR)
for i in range(len(gaussListR)):
    img=gaussListR[i][0]
    keys=sift_keypointsR[i]
    points = [sublist[0] for sublist in keys]
    draw_dots(img,points)
    

descList=getDescr(sift_keypoints,gaussList)

pcaDescr=doPca(descList)
print(pcaDescr)

allpoints=[]
img=gaussList[0][0]
for i in range(len(gaussList)):
    keys=sift_keypoints[i]
    points = [sublist[0] for sublist in keys]
    allpoints=allpoints+points
draw_dots(img,allpoints)


print(len(allpoints))



##############################  Problem 2 ############################# 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the custom CNN architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, pickle_file, transform=None):

        print(pickle_file)
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
		
        self.data = data_dict['data']
        self.labels = data_dict['labels']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.transpose(self.data[idx].reshape((3, 32, 32)), (1, 2, 0))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# # Define any additional transformations you need
transform = transforms.Compose([
    transforms.ToTensor(),

])

# Create the custom dataset
custom_dataset_train1 = CustomCIFAR10Dataset('data/train/data_batch_1', transform=transform)
custom_dataset_train2 = CustomCIFAR10Dataset('data/train/data_batch_2', transform=transform)
custom_dataset_train3 = CustomCIFAR10Dataset('data/train/data_batch_3', transform=transform)

# Create a data loader
batch_size = 64
train_data_loader1 = DataLoader(custom_dataset_train1, batch_size=batch_size, shuffle=True, num_workers=2)
train_data_loader2 = DataLoader(custom_dataset_train2, batch_size=batch_size, shuffle=True, num_workers=2)
train_data_loader3 = DataLoader(custom_dataset_train3, batch_size=batch_size, shuffle=True, num_workers=2)


custom_dataset_test = CustomCIFAR10Dataset('data/test/test_batch', transform=transform)
test_data_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)


# Initialize the model, loss function, and optimizer
model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_loss = []
test_loss = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_data_loader1:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    for inputs, labels in train_data_loader2:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    for inputs, labels in train_data_loader3:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss.append(running_loss/(len(train_data_loader1)+len(train_data_loader2)+len(train_data_loader3)))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/(len(train_data_loader1)+len(train_data_loader2)+len(train_data_loader3))}")

# Testing the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy}")

plt.figure()
plt.plot(train_loss)
plt.title('Loss vs epoch')
plt.xlabel('Epoch')
plt.savefig('loss_relu.png')





def evaluate_model(model, data, labels, transform=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(data)):
            image = data[i]

            if transform:
                image = transform(image)

            image = torch.unsqueeze(image, 0).to(device)
            outputs = model(image)

            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (predicted.item() == labels[i])

    accuracy = correct / total
    return accuracy


data = np.load('data/test/test_additional.npy', allow_pickle=True)
labels = np.load('data/test/labels.npy', allow_pickle=True)

print(data.shape, labels.shape)

transform = transforms.Compose([
    transforms.ToTensor(),
])


accuracy = evaluate_model(model, data, labels, transform=transform)

print(f"Accuracy on the dataset: {accuracy * 100:.2f}%")