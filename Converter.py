import cv2
import numpy as np
import math
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


def get_first_orde(img_gray):
    # Get All Frequency Total
    numAllFrq = len(img_gray) * len(img_gray[0])

    # Get histogram value form grayscale image
    hist = cv2.calcHist([img_gray], [0], None, [255], [0, 255])

    # Get Grayscale intensity Value
    url_set = []
    for row in img_gray:
        for col in row:
            if col not in url_set:
                url_set.append(col)
            else:
                pass

    # Get mean value
    mean = 0
    for item in url_set:
        mean += (item * int(hist[item]))
    mean = mean / numAllFrq

    # Get Median Value
    median = ndimage.median(img_gray, index=None)

    # Get Max Value
    temp_max = []
    for row in img_gray:
        temp_max.append(max(row))
    max_val = max(temp_max)

    # Get Min Value
    temp_min = []
    for row in img_gray:
        temp_min.append(min(row))
    min_val = min(temp_min)

    # Get Variance value
    var = 0
    for item in url_set:
        var += (pow((item - mean), 2) * (int(hist[item]) / numAllFrq))

    # Get Standard Deviance value
    std_dev = math.sqrt(var)

    # Get kurtois value
    kurtois = kurtosis(img_gray, axis=None, fisher=True, bias=False)

    # Get Entropy
    Entropy = 0
    for item in url_set:
        Entropy += ((int(hist[item]) / numAllFrq) * math.log(int(hist[item]) / numAllFrq))
    Entropy = Entropy * (-1)

    # Get contrast value
    Contrast = max_val - min_val

    # Get skewness value
    skewness = skew(img_gray, axis=None, bias=False)

    return [round(mean, 2), round(median, 2), round(max_val, 2), round(min_val, 2), round(var, 2), round(std_dev, 2),
            round(skewness, 2), round(kurtois, 2), round(Entropy, 2), Contrast]


def get_RGB(img_color):
    # Get RGB average value
    avg_color_per_row = np.average(img_color, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return avg_color


def second_order(img_gray):
    # Get graycomatrix value
    gcm = graycomatrix(img_gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    # Get ASM
    asm_val = graycoprops(gcm, 'ASM')

    # Get contrast
    contrast_val = graycoprops(gcm, 'contrast')

    # Get correlation
    correlation_val = graycoprops(gcm, 'correlation')

    # Get inverse difference moment
    idm_val = graycoprops(gcm, 'homogeneity')

    # Get Entropy
    entropy_val = []
    entropy_val.append(shannon_entropy(gcm[:, :, 0, 0]))
    entropy_val.append(shannon_entropy(gcm[:, :, 0, 1]))
    entropy_val.append(shannon_entropy(gcm[:, :, 0, 2]))
    entropy_val.append(shannon_entropy(gcm[:, :, 0, 3]))

    return [asm_val, contrast_val, correlation_val, idm_val, entropy_val]

def dfs_tabs(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)
    writer.save()