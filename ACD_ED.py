import glob
import os

import cv2
import torch
import torch.nn as nn
import numpy as np
import face_recognition

def Average_Content_Distance(real_path, gent_path):
    real_face = face_recognition.load_image_file(real_path)
    gent_face = face_recognition.load_image_file(gent_path)
    real_cont = face_recognition.face_encodings(real_face)
    gent_cont = face_recognition.face_encodings(gent_face)
    # if np.sum(np.array(real_cont)) != 0 and np.sum(np.array(gent_cont)) !=0:
    #     real_cont = real_cont[0]
    #     gent_cont = gent_cont[0]
    # dis = nn.MSELoss()(torch.tensor(gent_cont), torch.tensor(real_cont))
    if len(real_cont) == 0 or len(gent_cont) == 0:
        print(real_path, gent_path)
    else:
        real_cont = face_recognition.face_encodings(real_face)[0]
        gent_cont = face_recognition.face_encodings(gent_face)[0]
        # dis = face_recognition.face_distance(real_cont, gent_cont)
        dis = np.linalg.norm(real_cont - gent_cont)
        if isinstance(dis, float):
        # if dis is not None:
            return dis


def Expression_Distance(real_csv, fake_csv):
    real_content_txt = np.loadtxt(real_csv, float, delimiter=',', skiprows=1)
    real_content = real_content_txt[2:19]

    fake_content_txt = np.loadtxt(fake_csv, float, delimiter=',', skiprows=1)
    fake_content = fake_content_txt[2:19]

    dis = np.linalg.norm(real_content-fake_content)

    return dis


'''
calculating ACD
'''
if __name__ == '__main__':
    real_names = os.listdir(r'.\St_gan_affect_HaSa\xrec/')
    real_tmp = r'.\affect_HaSa_test_segface\imgs'
    # fake_tmp = r'.\affect_seg_face_xrec'
    # fake_tmp = r'.\St_gan_affect_HaSa\xrec'
    # fake_tmp = r'.\affect_HaSa_xrec'
    # fake_tmp = r'.\affect_HaSa_xrec'
    fake_tmp = r'.\affect_HaSa_xrec'
    # fake_tmp = r'.\affect_HaSa_xrec'
    all_dis = []
    for name in real_names:
        real_path = os.path.join(real_tmp, name)
        fake_path = os.path.join(fake_tmp, name[:-4]+'.png')
        dis = Average_Content_Distance(real_path, fake_path)
        all_dis.append(dis)
    all_dis = list(filter(None, all_dis))
    # ave_dis = sum(all_dis)/len(all_dis)
    # print(ave_dis)
    ave_dis = np.round(sum(all_dis)/len(all_dis), 3)
    print(ave_dis)

'''
calculating ED
'''
# import pickle
# # test_au = open(r'.\affect_HaSa_test_au.pkl', 'rb')
# test_au = open(r'.\affect_HaSa_test_segface_au.pkl', 'rb')
# test = pickle.load(test_au, encoding='iso-8859-1')
# # xrec_au = open(r'.\affect_HaSa_xrec_au\US_GAN_HaSa_test.pkl', 'rb')
# # xrec_au = open(r'.\affect_HaSa_xrec_au\SARGAN_HaSa_test.pkl', 'rb')
# xrec_au = open(r'.\affect_HaSa_xrec_au\MFS_HaSa_test.pkl', 'rb')
# # xrec_au = open(r'.\St_gan_affect_HaSa\au\St_gan_affect_HaSa_49.pkl', 'rb')
# xrec = pickle.load(xrec_au, encoding='iso-8859-1')
# vali_names = xrec.keys()
# all_dis = []
# for name in vali_names:
#     # name = '3754f0d37d135f342dda5b72da3814a91e9e78106e5b7cd95ec1a859'
#     x = xrec.get(name)
#     t = test.get(name)
#     # if x.dtype == 'float64' and t.dtype == 'float64':
#     # print(type(x))
#     # print(type(t))
#     # print(name)
#     # if type(x) != 'NoneType' and type(t) != 'NoneType' and type(x-t)!= 'NoneType' :
#     if t is not None and x is not None:
#     # if sum(x)!=0 and sum(t)!=0:
#         dis = np.linalg.norm(x - t)
#         all_dis.append(dis)
# all_dis = list(filter(None, all_dis))
# aver = np.round(np.mean(np.array(all_dis)), 3)
# print(aver)
