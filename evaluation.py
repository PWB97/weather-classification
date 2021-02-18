import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import os
import csv
import re
import json

import utils
import opts
from train import train_model, eval_model, eval_logits, eval_model_tta, eval_logits_tta
from model import resnet, resnext, resnext_wsl, vgg_bn, densenet, inception_v3, dpn, effnet
from dataloader import TestDataset, my_transform, test_transform
# from sync_batchnorm import convert_model

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
    elif opt.network == 'dpn':
        model = dpn(opt.classes, opt.layers)
    elif opt.network == 'effnet':
        model = effnet(opt.classes, opt.layers)
    # elif opt.network == 'pnasnet_m':
    #     model = pnasnet_m(opt.classes, opt.layers, opt.pretrained)

    # model = nn.DataParallel(model, device_ids=[4])
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = nn.DataParallel(model, device_ids=[opt.gpu_id, opt.gpu_id+1])
    # model = convert_model(model)
    model = model.to(device)

    images, names = utils.read_test_data(os.path.join(opt.root_dir, opt.test_dir))

    dict_= {}
    for crop_size in [opt.crop_size]:
        if opt.tta:
            transforms = test_transform(crop_size)
        else:
            transforms = my_transform(False, crop_size)

        dataset = TestDataset(images, names, transforms)

        loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=False, num_workers=4)
        state_dict = torch.load(
                    opt.model_dir+'/'+opt.network+'-'+str(opt.layers)+'-'+str(crop_size)+'_model.ckpt')
        if opt.network == 'densenet':
            pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        model.load_state_dict(state_dict)
        if opt.vote:
            if opt.tta:
                im_names, labels = eval_model_tta(loader, model, device=device)
            else:
                im_names, labels = eval_model(loader, model, device=device)
        else:
            if opt.tta:
                im_names, labels = eval_logits_tta(loader, model, device=device)
            else:
                im_names, labels = eval_logits(loader, model, device)
        im_labels = []
        # print(im_names)
        for name, label in zip(im_names, labels):
            if name in dict_:
                dict_[name].append(label)
            else:
                dict_[name] = [label]


    header = ['filename', 'type']
    utils.mkdir(opt.results_dir)
    result = opt.network + '-' +str(opt.layers) + '-'+str(opt.crop_size)+ '_result.csv'
    filename = os.path.join(opt.results_dir, result)
    with open(filename, 'w', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        for key in dict_.keys():
            v = np.argmax(np.sum(np.array(dict_[key]), axis=0)) + 1
            # v = list(np.sum(np.array(dict_[key]), axis=0))
            f_csv.writerow([key, v])

opt = opts.parse_args()
main(opt)







