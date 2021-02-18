import cv2
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
import csv
import glob
import os
import time
from multiprocessing import Pool

# import smote_variants as sv


ImageFile.LOAD_TRUNCATED_IMAGES = True

weather_classes = ['雨凇', '雾凇', '雾霾', '霜', '露', '结冰', '降雨', '降雪', '冰雹']
#                    1       2      3     4     5     6       7      8      9
def classes_num(filename):
    labels = [0]*9
    with open(filename) as f:
        f_csv = csv.reader(f)
        f_csv.__next__()
        for row in f_csv:
            labels[int(row[1])-1] += 1
    return labels

def image_info(dir):
    min_w, min_h = 1000, 1000
    max_w, max_h = 0, 0
    w, h = [], []
    ex = []
    for img_path in glob.glob(dir+'/*'):
        try:
            img = Image.open(img_path)
            if img.size[0] < min_w:
                min_w = img.size[0]
            elif img.size[0] > max_w:
                max_w = img.size[0]
            if img.size[1] < min_h:
                min_h = img.size[1]
            elif img.size[1] > max_h:
                max_h = img.size[1]
            w.append(img.size[0])
            h.append(img.size[1])

        except(OSError, NameError):
            ex.append(img_path)
            img = cv2.imread(img_path)
            print(img.shape)
    print("max_w:{}, max_h:{}".format(max_w, max_h))
    print("min_w:{}, min_h:{}".format(min_w, min_h))
    print('len:{}, mean_w:{}, mean_h:{}'.format(len(w), np.mean(w), np.mean(h)))
    print(ex)


def read_test_data(fdir):
    images = []
    im_names =[]
    i=0
    for img_path in tqdm(glob.glob(fdir+'/*')):
        im = load_image(os.path.join(fdir, img_path))
        im_names.append(img_path.split('/')[-1])
        images.append(im)
        i+=1
        if i>10000:
            break
    return images, im_names

def read_non_ice_snow_data(fdir, filename):
    images = []
    labels = []

    with open(filename) as f:
        f_csv = csv.reader(f)
        f_csv.__next__()
        print('loading image...')
        i=0
        for row in tqdm(f_csv):
            if row[0] == 'cad097b0899f45bcba277adf5344097e.png':
                continue
            if int(row[1]) in [6, 8]:
                continue
            elif int(row[1]) in [7]:
                labels.append(5)
            elif int(row[1]) in [9]:
                labels.append(6)
                # labels.append(int(row[1]) - 3)
            else:
                labels.append(int(row[1])-1)
            images.append(os.path.join(fdir, row[0]))

            i+=1
            if i>7000:
                break

    return images, labels

def read_ice_snow_data(fdir, filename):
    images = []
    labels = []
    # names = []
    # path = os.path.join(fdir, filename)

    with open(filename) as f:
        f_csv = csv.reader(f)
        f_csv.__next__()
        print('loading image...')
        i=0
        for row in tqdm(f_csv):
            if row[0] == 'cad097b0899f45bcba277adf5344097e.png':
                continue
            if int(row[1]) not in [6, 8]:
                continue
            # img = load_image(os.path.join(fdir, row[0]))
            images.append(os.path.join(fdir, row[0]))

            labels.append(int(int(row[1])-1==5))
            # names.append(row[0])
            i+=1
            if i>7000:
                break

    return images, labels

def read_data(fdir, filename, train_less=False, clean_data=False):
    images = []
    labels = []
    # names = []
    # path = os.path.join(fdir, filename)
    need_cleans = []
    if clean_data:
        with open('./err.csv', 'r') as f:
            f_csv = csv.reader(f)
            for row in tqdm(f_csv):
                need_cleans.append(row[0])

    with open(filename) as f:
        f_csv = csv.reader(f)
        f_csv.__next__()
        print('loading image...')
        i=0
        for row in tqdm(f_csv):
            if row[0] == 'cad097b0899f45bcba277adf5344097e.png':
                continue
            if clean_data:
                if row[0] in need_cleans:
                    continue
            if train_less:
                images.append(os.path.join(fdir, row[0]))
                if int(row[1]) in [6, 8]:
                    labels.append(0)
                # elif int(row[1]) in [8, 9]:
                #     labels.append(int(row[1])-2)
                else:
                    # labels.append(int(row[1])-1)
                    labels.append(1)

                continue

            # img = load_image(os.path.join(fdir, row[0]))
            images.append(os.path.join(fdir, row[0]))
            labels.append(int(row[1])-1)
            # names.append(row[0])
            i+=1
            # if i>7000:
            #     break

    return images, labels


def read_smote_data(fdir, filename, val_num=500):
    images = []
    labels = []
    # path = os.path.join(fdir, filename)
    with open(filename) as f:
        f_csv = csv.reader(f)
        f_csv.__next__()
        print('loading image...')
        for row in tqdm(f_csv):
            if row[0] == 'cad097b0899f45bcba277adf5344097e.png':
                continue
            images.append(os.path.join(fdir, row[0]))
            labels.append(int(row[1])-1)

    train_data = images[val_num:], labels[val_num:]
    val_data = images[:val_num], labels[:val_num]
    return train_data, val_data


# def smote_data(images, labels):
#     # images = images[:50]
#     # labels = labels[:50]
#     shape = np.shape(images)
#     nums = shape[0] // 2
#     oversampler = sv.MulticlassOversampling(sv.Borderline_SMOTE2(proportion=0.7, n_neighbors=3, k_neighbors=3, n_jobs=12)) # MDO
#     X, y = oversampler.sample(np.reshape(images, (len(images), -1)), labels)
#     X = X.reshape((len(y), shape[1], shape[2], shape[3])).astype(np.uint8)
#     mkdir('new_train')
#     with open('new_train_label.csv', 'a', encoding='utf-8') as f:
#         f_csv = csv.writer(f)
#         for i, x in enumerate(X):
#             im = Image.fromarray(x)
#             im.save('./new_train/'+str(i)+'.jpg', 'jpeg')
#             f_csv.writerow([str(i)+'.jpg', y[i]+1])
#
#     print('org: %d -> x: %d' % (len(labels), len(y)))
#     ys = [0]*10
#     for i in y:
#         ys[i+1] += 1
#     print(ys)

def load_image(filename):
    try:
        img = Image.open(filename)
    except(OSError, NameError):
        print('cv opened image')
        cv_img = cv2.imread(filename)
        img = Image.fromarray(cv_img)

    img = img.convert("RGB")
    # print(filename)
    return img

def load_image_label(params, resize=600):
    try:
        img = Image.open(params[0])
    except(OSError, NameError):
        print('cv opened image')
        cv_img = cv2.imread(params[0])
        img = Image.fromarray(cv_img)

    img = img.convert("RGB")
    img = img.resize((resize, resize), Image.ANTIALIAS)
    # print(filename)
    return np.array(img), params[1]

def to_tensor(data, dtype=torch.float16, device=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def mkdir(path):
    # give a path, create the folder
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def save_image(img, name):
    cv2.imwrite(name, img)

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


if __name__ == "__main__":
    file_dir = "/home/lzw/datasets/air"
    filename = "Train_label.csv"
    label_file = os.path.join(file_dir, filename)

    nums = classes_num(label_file)
    for i, cl in enumerate(weather_classes):
        print('{}:{}'.format(cl, nums[i]))
    print('nums:', np.sum(nums))

    image_info(os.path.join(file_dir,'train'))

    # train_data, val_data = read_smote_data(
    #     os.path.join(file_dir,'train'), os.path.join(file_dir,filename), val_num=0)
    # images, labels = train_data
    # tic = time.time()
    # pool = Pool(48)
    # img_names = pool.map(load_image_label, list(zip(images, labels)))
    # pool.close()
    # pool.join()
    # toc = time.time()
    # imgs = []
    # labs = []
    # for im, name in img_names:
    #     imgs.append(im)
    #     labs.append(name)
    # print('load image: ', toc-tic)
    # print(imgs[0].shape)

    # smote_data(imgs, labs)






















