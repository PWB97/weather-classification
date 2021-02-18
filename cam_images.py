import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import os
import csv

import utils
import opts
from train import grad_cam
from model import resnet, resnext, resnext_wsl, vgg_bn, densenet, inception_v3
from dataloader import TestDataset, my_transform

def main(opt):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(opt.gpu_id)
    else:
        device = torch.device('cpu')

    if opt.network == 'resnet':
        model = resnet(opt.classes, opt.layers)
    elif opt.network == 'resnext':
        model = resnext(opt.classes, opt.layers)
    elif opt.network == 'resnext_wsl':
        # resnext_wsl must specify the opt.battleneck_width parameter
        opt.network = 'resnext_wsl_32x' + str(opt.battleneck_width) +'d'
        model = resnext_wsl(opt.classes, opt.battleneck_width)
    elif opt.network == 'vgg':
        model = vgg_bn(opt.classes, opt.layers)
    elif opt.network == 'densenet':
        model = densenet(opt.classes, opt.layers)
    elif opt.network == 'inception_v3':
        model = inception_v3(opt.classes, opt.layers)

    model = nn.DataParallel(model, device_ids=[7, 6])
    model = model.to(device)

    train_data, _ = utils.read_data(
        os.path.join(opt.root_dir, opt.train_dir),
        os.path.join(opt.root_dir, opt.train_label),
        val_num=1)

    val_transforms = my_transform(False, opt.crop_size)
    dataset = WeatherDataset(train_data[0], train_data[1], val_transforms)

    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False, num_workers=2)

    model.load_state_dict(torch.load(
                opt.model_dir+'/'+opt.network+'-'+str(opt.layers)+'-'+str(crop_size)+'_model.ckpt'))



    im_labels = []
    for name, label in zip(im_names, labels):
        im_labels.append([name, label])

    header = ['filename', 'type']
    utils.mkdir(opt.results_dir)
    result = opt.network + '-' +str(opt.layers) + '-'+str(crop_size)+ '_result.csv'
    filename = os.path.join(opt.results_dir, result)
    with open(filename, 'w', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(im_labels)


opt = opts.parse_args()
main(opt)