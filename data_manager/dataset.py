import cv2
import numpy as np
import os
import torch.utils.data as data
from .transformer import  *
import torchvision.transforms as transforms
import random


class dataset_Aptos(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF

    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, self.DF.loc[index, 'id_code'])
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = cv2.imread(imgName)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF.loc[index, 'diagnosis']
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
            
        except:
            index = 0
            imgName = os.path.join(self.data_path, self.DF.loc[index, 'id_code'])
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = cv2.imread(imgName)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF.loc[index, 'diagnosis']
            label_onehot = np.zeros(5)
            label_onehot[label] = 1

        return Img, label, label_onehot

    def __len__(self):
        return len(self.DF)



class dataset_RFMiD(data.Dataset):
    def __init__(self, data_path, DF, transform = None):

        self.data_path = data_path
        self.transform = transform
        self.DF = DF


    def __getitem__(self, index):

        try:
            imgName = os.path.join(self.data_path, str(self.DF.loc[index, 'ID']))
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = cv2.imread(imgName)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)
            label = self.DF.loc[index, 'Disease_Risk']

            
        except:
            index = 0
            imgName = os.path.join(self.data_path, str(self.DF.loc[index, 'ID']))
            imgName = imgName + '.png'
            imgName = imgName.replace('\\', '/')

            Img = cv2.imread(imgName)
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
            Img = transforms.ToPILImage()(Img)

            if self.transform is not None:
                Img = self.transform(Img)

            # Get the Labels
            label = self.DF.loc[index, 'Disease_Risk']
            # label_onehot = np.zeros(2)
            # label_onehot[label] = 1

        return Img, label

    def __len__(self):
        return len(self.DF)

