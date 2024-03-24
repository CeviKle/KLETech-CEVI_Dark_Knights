import os
import sys
import time
import glob
import numpy as np
from pkg_resources import parse_requirements
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.nn as nn
from torch.autograd import Variable
from model import *
from loader import DataloaderSupervised, DataloaderSelfSupervised, NTIRELoaderCV2
import warnings
from tqdm import tqdm
from loss import LuminanceLoss, ContrastLoss, Vgg16, LossFunction
from vainF_ssim import MS_SSIM
import cv2
import pandas as pd

warnings.filterwarnings("ignore")

print("-----------------[INFO] Libraries Loaded-----------------")

parser = argparse.ArgumentParser()
 
parser.add_argument("-v", "--version", help='experiment version name')
parser.add_argument("-ct", "--continue_train", help='resume training from checkpoints', default=0)
parser.add_argument("-dl", "--data_loader", type=str, default='ss', required=True, help='s for supervised and ss for self supervised')
parser.add_argument("-bs", "--batch_size", type=int, default=8, help='defines batch size for training')
parser.add_argument("-e", "--epochs", type=int, default=1000, help='defines training epochs')
parser.add_argument('-lr',"--learning_rate", type=float, default=0.0003, help='learning rate')
parser.add_argument("-stg", "--stage", type=int, default=3, help='epochs')
parser.add_argument("-o", "--save", type=str, default='EXP', help='outputs paths')
parser.add_argument("--data_train",type=str, help='training data paths')
parser.add_argument("--data_test",type=str, help='testing data paths')

args = parser.parse_args()



args.save = args.save + '/' + args.version + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

device = torch.device("cuda")

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')



model = Network(stage=args.stage)

model.enhance.in_conv.apply(model.weights_init)
model.enhance.conv.apply(model.weights_init)
model.enhance.out_conv.apply(model.weights_init)
model.calibrate.in_conv.apply(model.weights_init)
model.calibrate.convs.apply(model.weights_init)
model.calibrate.out_conv.apply(model.weights_init)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=3e-4)
MB = utils.count_parameters_in_MB(model)
logging.info("model size = %s", MB)
print(MB)
# feature_loss = Vgg16().to(device)
# con_loss = ContrastLoss()
# lum_loss = LuminanceLoss()
# dssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)


if args.data_loader == 'ss':
    train_low_data_names = args.data_train
    TrainDataset = NTIRELoaderCV2(img_dir=train_low_data_names, task='train')

    test_low_data_names = args.data_test
    TestDataset = NTIRELoaderCV2(img_dir=test_low_data_names, task='test')

    train_loader = torch.utils.data.DataLoader(
        TrainDataset, 
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        TestDataset,
        batch_size= args.batch_size,
        num_workers=4,
        shuffle=True
    )
    total_step = 0

    idx = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []

        loop = tqdm(enumerate(train_loader), total=int(len(TrainDataset)/train_loader.batch_size))
        for idx, (inp, _) in loop:
            # idx+=1
            total_step += 1
            inp = inp.type(torch.FloatTensor)
            inp = Variable(inp, requires_grad=False).to(device)

            optimizer.zero_grad()
            # loss = model._loss(inp)
            loss = model._loss(inp)
            

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            # logging.info('train-epoch %03d %03d %f', epoch, idx, loss)
            utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))
            if epoch % 10 == 0 and total_step != 0:
                # logging.info('train %03d %f', epoch, loss)
                model.eval()
                with torch.no_grad():
                    for _, (input, image_name) in enumerate(test_loader):
                        input = Variable(input, volatile=True).cuda()
                        image_name = image_name[0].split('/')[-1].split('.')[0]

                        illu_list, ref_list, input_list, atten= model(input)

                        u_name = '%s.png' % (image_name + '_ref_' + str(epoch))
                        u_path = image_path + '/' + u_name
                        # saving intermediate outputs
                        o1_name = '%s.png' % (image_name + '_illu_' + str(epoch))
                        o1_path = image_path + '/' + o1_name
                        o3_name = '%s.png' % (image_name + '_inp_' + str(epoch))
                        o3_path = image_path + '/' + o3_name
                        o4_name = '%s.png' % (image_name + '_attn_' + str(epoch))
                        o4_path = image_path + '/' + o4_name

                        save_images(ref_list[0], u_path)
                        # save_images(illu_list[0], o1_path)
                        save_images(input_list[0], o3_path)
                        # save_images(atten[0], o4_path)
