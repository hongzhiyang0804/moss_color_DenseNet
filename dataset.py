import numpy as np
import os
from keras.utils import to_categorical
from PIL import Image
import random
# 载入数据路径
def load_satetile_image(input_dir, dataset='train'):
    label_list = []
    image_path = []
    # train处理
    if dataset == 'train':
        path = input_dir
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            last_name = child_path.split('/')
            child_path_last = last_name[-1]
            # print(child_path_last)
            # 获取各类别图片路径并载入矩阵,同时赋予相应的标签
            if child_path_last == 'mossyellow':
                dir_counter = 0
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'mosswhite':
                dir_counter = 1
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
        # 随机打散不同类别图片路径及对应标签
        cc = list(zip(image_path, label_list))
        random.shuffle(cc)
        image_path[:], label_list[:] = zip(*cc)
    # valid处理
    else:
        path = input_dir
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            last_name = child_path.split('/')
            child_path_last = last_name[-1]
            # print(child_path_last)
            # 获取各类别图片路径并载入矩阵,同时赋予相应的标签
            if child_path_last == 'mossyellow':
                dir_counter = 0
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'mosswhite':
                dir_counter = 1
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
        # 随机打散不同类别图片路径及对应标签
        cc = list(zip(image_path, label_list))
        random.shuffle(cc)
        image_path[:], label_list[:] = zip(*cc)
    return image_path, label_list
# 批量读取图片及对应标签
def batch_image(image_path, label_list, batch_size=64,index=0):
    img_list = []
    # 获取每批次下图片以及对应标签
    for j in image_path[index*batch_size: (index+1)*batch_size]:
        image = Image.open(j)
        # 图片缩放
        image = image.resize((416, 416))
        img = np.array(image)
        img = img / 255.0
        img_list.append(img)
    # 得到每批次的标签
    batch_size_label_list = label_list[index * batch_size:(index+1) * batch_size]
    # print(img_list)
    # 得到每批次图片
    x_batch = np.array(img_list)
    # print(x_batch.shape)
    # 标签转为one-hot
    y_batch = to_categorical(batch_size_label_list, 2)
    # print(y_batch.shape)
    return x_batch, y_batch

# image_pats_image(image_path, label_list, batch_size=8, index=i)



