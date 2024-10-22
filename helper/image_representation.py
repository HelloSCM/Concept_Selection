import os
import argparse


parser = argparse.ArgumentParser(description='Labeling Concepts via CLIP ViT-L/14.')
parser.add_argument('--data_name', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'lad_a', 'lad_f', 'lad_v', 'lad_e', 'lad_h', 'tiny_imagenet'])
parser.add_argument('--data_path', type=str, default='/data/scm22/')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


import pickle
import torch
import clip
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from dataset.lad import LAD
from dataset.tiny_imagenet import TinyImageNet


torch.set_num_threads(4)

model, preprocess = clip.load('ViT-L/14', device='cuda')
if args.data_name == 'cifar10':
    dataset = CIFAR10(root=args.data_path, train=False, download=False, transform=preprocess)
elif args.data_name == 'cifar100':
    dataset = CIFAR100(root=args.data_path, train=False, download=False, transform=preprocess)
elif args.data_name[:3] == 'lad':
    dataset = LAD(root=args.data_path, data_type=args.data_name[-1].upper(), transform=preprocess, mode='test', concept_bottleneck=False)
elif args.data_name == 'tiny_imagenet':
    dataset = TinyImageNet(root=args.data_path, transform=preprocess, mode='test')

Img_Repre_Mat = []

model.eval()
with torch.no_grad():
    for i in range(len(dataset)):
        image = dataset[i][0].unsqueeze(0).cuda()
        image_feature = model.encode_image(image).squeeze()
        Img_Repre_Mat.append(image_feature)

        if i % 10 == 0:
            print(f"Finish Extracting {i} Images!")

Img_Repre_Mat = np.array(torch.stack(Img_Repre_Mat).cpu())

with open(f'concept_bank/{args.data_name}/{args.data_name}_image_representation_test.pkl', 'wb') as file:
    pickle.dump(Img_Repre_Mat, file)