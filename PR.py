import glob
import os
from scipy.stats import entropy
import numpy as np
from skimage.feature import hog,local_binary_pattern
import cv2
import pickle
import matplotlib.pyplot as plt

def read_image(path):
    img_bgr = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, 'gray')
    # plt.show()
    return img_gray

def get_hog(img_gray):
    # hog_featue = hog(img_gray, orientations=180, pixels_per_cell=(112, 112),
    #                  cells_per_block=(1, 1), block_norm='L1', visualize=False)
    hog_featue = hog(img_gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L1', visualize=False)

    return hog_featue

def get_lbp(img_gray):
    lbp_img = local_binary_pattern(img_gray, 8, 1.0)
    max_bins = int(lbp_img.max() + 1)
    hist, _ = np.histogram(lbp_img, density=False, bins=max_bins, range=(0, max_bins))
    return hist


def get_all_hog(img_names, root_path, mask_root_path):
    all_hog = []
    for name in img_names:
        if os.path.exists(os.path.join(root_path, name+'.jpg')):
            img = read_image(os.path.join(root_path, name+'.jpg'))
            mask_img = read_image(os.path.join(mask_root_path, name+'.jpg'))
            hog_feature = get_hog(img)
            mask_hog_feature = get_hog(mask_img)
            # hog_feature = get_lbp(img)
            # mask_hog_feature = get_lbp(mask_img)
            all_hog.append(hog_feature-mask_hog_feature)
    hog_array = np.array(all_hog)
    # hog_norm = hog_array / (np.sum(np.abs(hog_array)) + 1e-5)
    hog_norm = np.sum(hog_array, axis=0) / (np.sum(np.abs(hog_array)))
    return hog_norm

def kl_dis(s_hog, sc_hog):
    KL = entropy(s_hog, sc_hog)
    return KL

if __name__ == '__main__':
    # csv_path = glob.glob(r'./affect_seg_face_test_au/*.csv')
    # for path in csv_path:
    #     content = np.loadtxt(path, float, delimiter=',', skiprows=1)
    #     au_float = content[2:19]
    #     s_names = []
    #     sc_names = []
    #     for a in range(17):
    #         if au_float[a]!=0:
    #             s_names.append()
    fr = open(r".\Affect\affect_HaSa_test_au.pkl", "rb")
    result = pickle.load(fr)
    all_names = np.array(list(result.keys()))
    au_float = np.array(list(result.values()))
    # root_path = r'.\affect_seg_face_mask_input'  # mask_input_images
    # root_path = r'.\affect_seg_face_test\imgs'     # segment_face_images
    # root_path = r'.\affect_seg_face_xrec'          # xrec_img
    root_path = r'.\St_gan_affect_HaSa\Mask_input'  # xrec_img
    mask_root_path = r'.\St_gan_affect_HaSa\mask'
    KL = []
    for i in range(6):
        s_name = all_names[np.where(au_float[:, i] != 0)]
        sc_name = all_names[np.where(au_float[:, i] == 0)]
        s_norm = get_all_hog(s_name, root_path, mask_root_path)
        s_norm = np.where(s_norm > 0, s_norm, 0)
        sc_norm = get_all_hog(sc_name, root_path, mask_root_path)
        sc_norm = np.where(sc_norm > 0, s_norm, 0)
        kl = kl_dis(s_norm+1e-10, sc_norm+1e-10)
        # kl = kl_dis(s_norm, sc_norm)
        KL.append(kl)
    print(KL)
