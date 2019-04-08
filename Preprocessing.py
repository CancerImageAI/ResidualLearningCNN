# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:22:16 2019

@author: Jing Gong
@e-mail: gongjing1990@163.com
@Fudan University Shanghai Cancer Center
"""

import SimpleITK as sitk
import numpy as np
import scipy
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from Data_Augmentation import img_augmentation


def readDCM_Img(FilePath):
    img = {}
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    Spacing = image.GetSpacing()
#    Origin = image.GetOrigin()
    img_array = img_array.transpose(2,1,0)
    img['array'] = img_array
    img['Spacing'] = np.array(Spacing)
#    img['Origin'] = Origin
    return img
    
def normalize_hu(image):
	#将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image*255
    return image.astype('uint8')

def resample(img, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    image = img['array']
    spacing = img['Spacing']
    img_size = np.array(image.shape)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, img_size, real_resize_factor

def crop_roi(image, seed_pos, img_size, resize_factor):
    initial_seed = [seed_pos[0], seed_pos[1], img_size[2]-seed_pos[2]]
    trans_seed = initial_seed*resize_factor
    if trans_seed[0]>32:
        x_min = int(trans_seed[0]-32)
        x_max = int(trans_seed[0]+32)
        delta_x = x_max-x_min-64
        x_max = x_max-delta_x
    else:
        x_min = 0
        x_max = 64
        
    if  trans_seed[1]>32:
        y_min = int(trans_seed[1]-32)
        y_max = int(trans_seed[1]+32)
        delta_y = y_max-y_min-64
        y_max = y_max-delta_y   
    else:
        x_min = 0
        y_max = 64
    
    if trans_seed[2]>32:
        z_min = int(trans_seed[2]-32)
        z_max = int(trans_seed[2]+32)
        delta_z = z_max-z_min-64
        z_max = z_max-delta_z
    else:
        z_min = 0
        z_max = 64
    roi = image[x_min:x_max, y_min:y_max, z_min:z_max]        
    return roi

if __name__ == '__main__':
    ListPath = r'.\GGO_data\GGO_List.csv' ## The  file info list of GGO
    ListFile = open(ListPath)
    List = pd.read_csv(ListFile)
    List_Num = List['Num'].tolist() ## The file name of each DICOM File
    X = List['X'].tolist()
    Y = List['Y'].tolist()
    Z = List['Z'].tolist()
    Type = List['Type'].tolist()
    Histopathology = List['Histopathology'].tolist()
    Class = List['Class'].tolist()
    size = List['Diameter'].tolist()
    IAC_num = 0
    non_IAC_num = 0
    IAC = []
    non_IAC = []
    for i in range(len(List_Num)):#
        if Type[i] != 'Solid' and Class[i] == 1:
            dcm_File = List_Num[i][0:7]        
            FilePath = './GGO_data/'+dcm_File
            img = readDCM_Img(FilePath)
            image, img_size, resize_factor = resample(img)
            image = normalize_hu(image)
            seed_pos = [X[i], Y[i], Z[i]]
            delta_step = [[-3,-3,0],[-3,0,0],[0,-3,0],[0,0,-3],
                          [-2,-2,0,],[-2,0,0],[0,-2,0],[0,0,-2],
                          [-1,-1,0],[-1,0,0],[0,-1,0],[0,0,-1],[-1,-1,-1],
                          [0,0,0],[1,1,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1],
                          [2,2,0],[2,0,0],[0,2,0],[0,0,2],
                          [3,3,0],[3,0,0],[0,3,0],[0,0,3]]
            for d in range(len(delta_step)):                    
                seed_pos1 = np.array(seed_pos)+np.array(delta_step[d])
                seed_pos1 = seed_pos1.tolist()             
                roi = crop_roi(image, seed_pos1, img_size, resize_factor)
                roi_rgb = np.array([roi[32,:,:],roi[:,32,:],roi[:,:,32]])
                roi_rgb = roi_rgb.transpose(2,1,0)
                if delta_step[d] == [0,0,0]: 
                    if Histopathology[i] == 'IAC':
                        img = img_augmentation(roi_rgb,prob=1)
                        for Img_type,roi_img in img.items():
                            if roi_img != []:
                                IAC_info = {}
                                IAC_num = IAC_num+1
                                Save_Name = 'IAC_'+str(IAC_num)
                                    
                                SavePath = os.path.join('./IAC_VS_pre/Crop_Img/IAC',Save_Name+'.jpg')
                                imageio.imwrite(SavePath,roi_img)
                                IAC_info['Class'] = Class[i]
                                IAC_info['Histopathology'] = Histopathology[i]
                                IAC_info['Num'] = Save_Name
                                IAC.append(IAC_info)
                        
                    elif Histopathology[i] != 'IAC':
                        img = img_augmentation(roi_rgb, prob = 0.4)
                        for Img_type,roi_img in img.items():
                            if roi_img != []:
                                non_IAC_info = {}
                                non_IAC_num = non_IAC_num+1
                                Save_Name = 'non_IAC_'+str(non_IAC_num)
                                
                                SavePath = os.path.join('./IAC_VS_pre/Crop_Img/non_IAC',Save_Name+'.jpg')
                                imageio.imwrite(SavePath,roi_img)
                                non_IAC_info['Class'] = Class[i]
                                non_IAC_info['Histopathology'] = Histopathology[i]
                                non_IAC_info['Num'] = Save_Name
                                non_IAC.append(non_IAC_info)
                else:
                    if Histopathology[i] == 'IAC':
                        IAC_info = {}
                        IAC_num = IAC_num+1
                        Save_Name = 'IAC_'+str(IAC_num)
                            
                        SavePath = os.path.join('./IAC_VS_pre/Crop_Img/IAC',Save_Name+'.jpg')
                        imageio.imwrite(SavePath,roi_rgb)
                        IAC_info['Class'] = Class[i]
                        IAC_info['Histopathology'] = Histopathology[i]
                        IAC_info['Num'] = Save_Name
                        IAC.append(IAC_info)
                    elif Histopathology[i] != 'IAC':
                        generator = np.random.choice([True, False],1,p=[0.3, 1-0.3])
                        if generator: 
                            non_IAC_info = {}
                            non_IAC_num = non_IAC_num+1
                            Save_Name = 'non_IAC_'+str(non_IAC_num)
                            
                            SavePath = os.path.join('./IAC_VS_pre/Crop_Img/non_IAC',Save_Name+'.jpg')
                            imageio.imwrite(SavePath,roi_rgb)
                            non_IAC_info['Class'] = Class[i]
                            non_IAC_info['Histopathology'] = Histopathology[i]
                            non_IAC_info['Num'] = Save_Name
                            non_IAC.append(non_IAC_info)

                                
   Info = IAC+non_IAC
   df = pd.DataFrame(Info).fillna('null')
   df.to_csv('D:/LungCancer/GGO_DataSet/IAC_VS_pre/Crop_Img/List_Info.csv',index=False,sep=',')

    
