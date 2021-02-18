import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from tqdm import tqdm
import numpy as np
import copy

import utils
from grad_cam import GradCam, show_cam_on_image
from dataloader import UnNormalize
from tta import *

tta_list = [NoneAug(),
            Hflip(),
            # Vflip(),
            Rotate(30),
            Rotate(60),
            # Rotate(90),
            ]

def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(dataloaders, model, criterion, optimizer, summary_writer,
        scheduler=None, scheduler_name='multistep', num_epochs=20, device=None,
        is_inception=False, mixup=False, alpha=1.0, val_dis=[]):
    tic = time.time()

    acc_history = []
    best_acc = 0
    best_model_wgt = None

    step_per_epoch = len(dataloaders['train'].dataset) / dataloaders['train'].batch_size

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                running_loss = 0.0
            else:
                model.eval()
            running_correct = 0.0
            err = []
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)


                if phase == 'train' and mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha, device)

                if phase == 'train' and is_inception:
                    logits, aux_logits = model(inputs)
                    if mixup:
                        loss1 = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                        loss2 = mixup_criterion(criterion, aux_logits, targets_a, targets_b, lam)
                        loss = (loss1 + 0.4*loss2)
                    else:
                        loss1 = criterion(logits, labels)
                        loss2 = criterion(aux_logits, labels)
                        loss = (loss1 + 0.4*loss2)
                else:
                    if phase == 'train' and mixup:
                        logits = model(inputs)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        with torch.no_grad():
                            logits = model(inputs)
                            loss = criterion(logits, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    summary_writer.add_scalar(tag='loss', scalar_value=loss.item(),
                                        global_step=step_per_epoch*epoch+i)

                    running_loss += loss.item()
                    if scheduler is not None and (scheduler_name == 'cycle' or
                                                  scheduler_name == 'warmup' or
                                                  scheduler_name == 'cos' or
                                                  scheduler_name == 'cosw'):
                        scheduler.step()

                _, preds = logits.max(1)
                if phase == 'train' and mixup:
                    correct = (lam * preds.eq(targets_a.data).sum().float()
                                + (1 - lam) * preds.eq(targets_b.data).sum().float())
                else:
                    correct = (preds == labels).sum()

                running_correct += correct
                if phase == 'val':
                    # print('label:', labels)
                    # print('preds:', preds)
                    error_label = (preds != labels).long() * (labels + 1)
                    error_label = error_label[error_label>0]
                    err.append(error_label)
                
                del inputs, labels, loss
                torch.cuda.empty_cache()

            epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                print('train --> loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
            else:
                print('val --> acc:{:.4f}'.format(epoch_acc))
                if len(val_dis) >= 8:
                    errors = torch.cat(err, 0)
                    # total_err = errors.cpu().numpy().shape[0]
                    err_rate = np.bincount(errors.cpu().numpy(), minlength=9)[1:] / val_dis
                    [print('{}: {:.2f}'.format(utils.weather_classes[i], rate)) for i, rate in enumerate(err_rate)]

                acc_history.append(epoch_acc.item())
                summary_writer.add_scalar(tag='correct', scalar_value=epoch_acc, global_step=epoch)
                summary_writer.add_scalar(tag='learning rate', scalar_value=optimizer.param_groups[0]["lr"],
                                          global_step=epoch)
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    if epoch > num_epochs//2:
                        best_model_wgt = copy.deepcopy(model.state_dict())
        print()
        if scheduler is not None:
            if scheduler_name == 'multistep':
                scheduler.step()
            elif scheduler_name == 'plateau':
                scheduler.step(epoch_loss)

    summary_writer.close()
    toc = time.time()
    time_elapsed = toc - tic
    print('training time -> %d:%.2f' % (time_elapsed // 60, time_elapsed % 60))
    print('best_acc:', best_acc.item())

    if best_model_wgt is not None:
        model.load_state_dict(best_model_wgt)

    return model, best_acc.item()

def clean_data(loader, model, device):
    model.eval()
    labels = []
    im_names = []
    err = []
    not_correct = 0
    for images, labels, names in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        logits = torch.softmax(model(images), 1)

        score, preds = logits.max(1)
        not_correct += (preds != labels).sum()
        # score_ = ((score>=0.9) | (0.4<score) * (score<0.6)).long()
        error_label = (preds != labels).long() * (labels + 1) # *score_
        for i, lab in enumerate(error_label):

            # if lab>0 and (preds[i] in [5, 7] or labels[i] in [5, 7]):
            if lab>0:
                im_names.append((names[i].split('/')[-1], preds[i].item()+1, labels[i].item()+1, score[i].item()))

    print("%d/%d" % (len(im_names), not_correct))
    return im_names


def eval_model(loader, model, device):
    model.eval()
    labels = []
    im_names = []
    for images, names in tqdm(loader):
        images = images.to(device)
        logits = model(images)

        _, preds = logits.max(1)
        for p, n in zip(preds.cpu().numpy().tolist(), names):
            im_names.append(n)
            labels.append(p+1)
    return im_names, labels


def eval_logits(loader, model, device=None):
    model.eval()
    labels = []
    im_names = []
    for images, names in tqdm(loader):
        images = images.to(device)
        logits = model(images)

        for p, n in zip(logits.detach().cpu().numpy().tolist(), names):
            im_names.append(n)
            labels.append(p)
    return im_names, labels

def eval_model_tta(loader, model, tta_augs=tta_list, device=None):
    model.eval()
    labels = []
    im_names = []
    for images, names in tqdm(loader):
        images = TensorToPILs(images)
        logits = []
        for aug in tta_augs:
            aug_imgs = PILsToTensor(aug(images)).to(device)
            outputs = model(aug_imgs).detach().cpu().numpy().tolist()
            logits.append(outputs)
        # print(np.shape(logits))
        logits = np.mean(np.array(logits), axis=0)

        preds = np.argmax(logits, axis=1)
        for p, n in zip(preds, names):
            im_names.append(n)
            labels.append(p+1)
    return im_names, labels

def eval_logits_tta(loader, model, tta_augs=tta_list, device=None):
    model.eval()
    labels = []
    im_names = []
    for images, names in tqdm(loader):
        images = TensorToPILs(images)
        logits = []
        for aug in tta_augs:
            aug_imgs = PILsToTensor(aug(images)).to(device)
            outputs = model(aug_imgs).detach().cpu().numpy().tolist()
            logits.append(outputs)
        # print(np.shape(logits))
        logits = np.mean(np.array(logits), axis=0)

        for p, n in zip(logits, names):
            im_names.append(n)
            labels.append(p)
    return im_names, labels


def grad_cam(loader, model, device):
    model.eval()
    labels = []
    im_names = []
    use_cuda = device != None
    model_no_fc = copy.deepcopy(model)
    del model_no_fc.fc
    cam = GradCam(model=model_no_fc, target_layer_names=['layer4'], use_cuda=use_cuda, org_model=model)
    utils.mkdir('cam')
    unorm = UnNormalize()
    for i, (images, labels, org_imgs) in enumerate(tqdm(loader)):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        _, preds = logits.max(1)

        right = (preds == labels)
        masks = cam(images, None)
        # print(org_imgs.shape)
        if right:
            name = './cam/{}-{}-{}-{}.jpg'.format(i, utils.weather_classes[preds[0]], utils.weather_classes[labels[0]], right[0])
            img = unorm(images[0]).cpu().numpy().transpose(1, 2, 0)
            show_cam_on_image(img, masks, name)
            # print(org_imgs[0].shape)
            utils.save_image(org_imgs[0].numpy(), './cam/{}-{}.jpg'.format(i, utils.weather_classes[labels[0]]))

    print('cam finished!')













