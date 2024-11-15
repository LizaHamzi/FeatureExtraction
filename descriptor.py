import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo
def glcm(data):
   co_matrix = graycomatrix(data, [1], [0], symmetric=True, normed=True)
   dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
   contrast = graycoprops(co_matrix, 'contrast')[0, 0]
   correlation = graycoprops(co_matrix, 'correlation')[0, 0]
   energy = graycoprops(co_matrix, 'energy')[0, 0]
   asm = graycoprops(co_matrix, 'ASM')[0, 0]
   homogeneity = graycoprops(co_matrix, 'homogeneity')[0, 0]
   features = [np.float32(dissimilarity), np.float32(contrast), np.float32(correlation), np.float32(energy), np.float32(asm), np.float32(homogeneity)]
   return features
def bitdesc(data):
   features = bio_taxo(data)
   features = [np.float32(feature) for feature in features]
   required_length = 14
   if len(features) < required_length:
       features += [np.float32(0)] * (required_length - len(features))
   return features
def haralick_feat(data):
   features = haralick(data).mean(0).tolist()
   return features
def bit_glcm_haralick(data):
   return bitdesc(data) + glcm(data) + haralick_feat(data)

def features_extraction_concat(image_path):
   try:
       rgb = cv2.imread(image_path)
       if rgb is None:
           raise ValueError("Failed to read image. Please check the image path and format.")
       r, g, b = cv2.split(rgb)
       r_gray = r if len(r.shape) == 2 else cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
       g_gray = g if len(g.shape) == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
       b_gray = b if len(b.shape) == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
       gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
       r_features = glcm(r_gray)
       g_features = bitdesc(g_gray)
       b_features = bitdesc(b_gray)
       rgb_features = bit_glcm_haralick(gray_rgb)
       return r_features + g_features + b_features + rgb_features
   except Exception as e:
       print(f'Split error: {e}')
       return []
   


def glcm_beta(image_path):
    data = cv2.imread(image_path, 0)
    co_matrix = graycomatrix(data, [1], [np.pi/4], None, symmetric=False, normed=False)
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    features = [np.float32(dissimilarity), np.float32(cont), np.float32(corr), np.float32(ener), np.float32(asm), np.float32(homo)]
    print(f"GLCM features shape: {len(features)}")
    return features

def bitdesc_beta(image_path):
    data = cv2.imread(image_path, 0)
    features = bio_taxo(data)
    features = [np.float32(feature) for feature in features]
    required_length = 14  
    if len(features) < required_length:
        features += [np.float32(0)] * (required_length - len(features))
    print(f"BIT features shape (after padding): {len(features)}")
    return features

def haralick_feat_beta(image_path):
    data= cv2.imread(image_path,0)
    print(f"Haralick features shape: {len(data)}")
    return haralick(data).mean(0).tolist()