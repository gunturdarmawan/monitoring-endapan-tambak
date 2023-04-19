import datetime

import cv2
import pandas as pd
from absl import flags
from absl.flags import FLAGS

import Converter

flags.DEFINE_string('image', None, 'path to image file')
flags.DEFINE_string('folder', None, 'path to folder')


def image_processing(image):
    global df_rgb, df_firstOrder, df_secondOrder
    img = cv2.imread(image)
    img_color = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

    first_order = Converter.get_first_orde(img_gray)
    second_order = Converter.second_order(img_gray)
    rgb = Converter.get_RGB(img_color)

    data_rgb = {
        'Red': rgb[0],
        'Green': rgb[1],
        'Blue': rgb[2],
    }
    data_first_order = {
        'Mean': first_order[0],
        'Median': int(first_order[1]),
        'Max': int(first_order[2]),
        'Min': int(first_order[3]),
        'Variance': first_order[4],
        'Standard Deviasi': first_order[5],
        'Skewness': first_order[6],
        'Kurtois': first_order[7],
        'Entropy': first_order[8],
        'Contrast': int(first_order[9]),
    }
    data_second_order = {
        'ASM 0': round(second_order[0][0][0], 5),
        'ASM 45': round(second_order[0][0][1], 5),
        'ASM 90': round(second_order[0][0][2], 5),
        'ASM 135': round(second_order[0][0][3], 5),
        'Contrast 0': round(second_order[1][0][0], 5),
        'Contrast 45': round(second_order[1][0][1], 5),
        'Contrast 90': round(second_order[1][0][2], 5),
        'Contrast 135': round(second_order[1][0][3], 5),
        'Correlation 0': round(second_order[2][0][0], 5),
        'Correlation 45': round(second_order[2][0][1], 5),
        'Correlation 90': round(second_order[2][0][2], 5),
        'Correlation 135': round(second_order[2][0][3], 5),
        'IDM 0': round(second_order[3][0][0], 5),
        'IDM 45': round(second_order[3][0][1], 5),
        'IDM 90': round(second_order[3][0][2], 5),
        'IDM 135': round(second_order[3][0][3], 5),
        'Entropy 0': round(second_order[4][0], 5),
        'Entropy 45': round(second_order[4][1], 5),
        'Entropy 90': round(second_order[4][2], 5),
        'Entropy 135': round(second_order[4][3], 5),
    }
    df_rgb = pd.DataFrame(data=[data_rgb])
    df_firstOrder = pd.DataFrame(data=[data_first_order])
    df_secondOrder = pd.DataFrame(data=[data_second_order])

    # list of dataframes and file names
    dfs = [df_rgb, df_firstOrder, df_secondOrder]
    filenames = ['rgb.csv', 'first_order.csv', 'second_order.csv']

    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i in range(len(dfs)):
        df = dfs[i]
        filename = filenames[i]
        file_path = "static/result/" + filename
        df.to_csv(file_path, index=False)

    for i in range(len(dfs)):
        df = dfs[i]
        filename = timestamp + "_" + filenames[i]
        file_path = "static/result/" + filename
        df.to_csv(file_path, index=False)

    return filenames
