import glob
import os
import pandas as pd
import cv2
import numpy as np
import random
import shutil
from PIL import Image

def creat_folder(curret_path):
    if not os.path.exists(curret_path):
        os.makedirs(curret_path)


'''
collecting  pkl files
'''
import pickle
from tqdm import tqdm
import glob
import os
def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        # pickle.dump(data, f)

csv_pathes = glob.glob(r'.\Affect\affect_HaSa_test_segface_landmark\*.csv')
# csv_pathes = glob.glob(r'.\dataset\Ablation_Study\St_gan_affect_HaSa\au\*.csv')
data = dict()
for path in csv_pathes:
    name = os.path.basename(path[:-4])
    content = np.loadtxt(path, float, delimiter=',', skiprows=1)
    data[name] = np.array([content[2], content[4], content[6], content[10], content[12], content[16]])
# save_dict(data, r'.\Ablation_Study\St_gan_affect_HaSa\au\St_gan_affect_HaSa_49')
save_dict(data, r'.\Affect\affect_HaSa_test_segface_landmark\affect_HaSa_test_segface_landmark')

'''
collecting segment out face
'''
from PIL import Image
save_tmp = r'.\Affect\affect_HaSa_test_segface'
pathes = glob.glob(r'.\Affect\affect_HaSa_test_au/*.csv')
data = dict()
for path in pathes:
    img_name = os.path.basename(path[:-4])
    content = np.loadtxt(path, float, delimiter=',', skiprows=1)
    # data[os.path.basename(path[:-4])] = [content[2:19]]
    img_pathes = os.listdir(os.path.join(r'.\Affect\affect_HaSa_test_au/', img_name+'_aligned'))
    if len(content.shape) == 1 and len(img_pathes) == 1:
        data[os.path.basename(path[:-4])] = np.array([content[2], content[4], content[6], content[10], content[12], content[16]])
        path = img_pathes[0]
        read_path = os.path.join(r'.\Affect\affect_HaSa_test_au/', img_name+'_aligned', path)
        save_path = os.path.join(save_tmp, img_name + '.jpg')
        im = Image.open(read_path)
        im.save(save_path)
        # shutil.copyfile(read_path, save_path)
        # os.remove(read_path)

'''Allocating traing and testing datasets'''
def collect_file(read_path, save_path):
    for x, classes, y in os.walk(read_path):
        for c in classes:
            image_names_all = os.listdir(os.path.join(read_path, c))
            file_num = len(image_names_all)
            # pick_num = int(file_num*1./9.)      # test dataset
            pick_num = int(file_num*0.8)      # test dataset
            selct_num = random.sample(image_names_all, pick_num)

            save_calss = os.path.join(save_path,c)
            creat_folder(save_calss)
            for img_name in selct_num:
                oo_path = os.path.join(read_path, c, img_name)
                ss_path = os.path.join(save_calss, img_name)
                shutil.move(oo_path, ss_path)

read_path = r'./AffectNet/'
save_path =r'.\AffectNet\train/'
collect_file(read_path, save_path)