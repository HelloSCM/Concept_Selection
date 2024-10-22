import os
import argparse


parser = argparse.ArgumentParser(description='Labeling Concepts via CLIP ViT-L/14.')
parser.add_argument('--data_name', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'lad_a', 'lad_f', 'lad_v', 'lad_e', 'lad_h', 'tiny-imagenet'])
parser.add_argument('--data_path', type=str, default='/data/.../')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


import pickle
import numpy as np
import torch
import clip
from torchvision.datasets import CIFAR10, CIFAR100
from dataset.lad import LAD
from dataset.tiny_imagenet import TinyImageNet


torch.set_num_threads(4)

model, preprocess = clip.load('ViT-L/14', device='cuda')
if args.data_name == 'cifar10':
    dataset = CIFAR10(root=args.data_path, train=True, download=False, transform=preprocess)
elif args.data_name == 'cifar100':
    dataset = CIFAR100(root=args.data_path, train=True, download=False, transform=preprocess)
elif args.data_name[:3] == 'lad':
    dataset = LAD(root=args.data_path, data_type=args.data_name[-1].upper(), transform=preprocess, mode='train', concept_bottleneck=False)
elif args.data_name == 'tiny-imagenet':
    dataset = TinyImageNet(root=args.data_path, transform=preprocess, mode='train')

with open(f'concept_bank/concept_bank_num_13933.pkl', 'rb') as file:
    concept_bank = pickle.load(file)
concept_embeddings = torch.from_numpy(np.array(list(concept_bank.values())))

Concept_Labels_org = []
Concept_Labels_norm = []

for i in range(len(dataset)):
    image = dataset[i][0].unsqueeze(0).cuda()
    image_feature_org = model.encode_image(image).squeeze()
    image_feature_norm = image_feature_org / image_feature_org.norm(dim=-1, keepdim=True)
    concept_labels_org = torch.zeros(concept_embeddings.shape[0])
    concept_labels_norm = torch.zeros(concept_embeddings.shape[0])
    for cpt in range(concept_embeddings.shape[0]):
        concept_embedding = concept_embeddings[cpt].cuda()
        concept_labels_org[cpt] = torch.matmul(image_feature_org, concept_embedding).cpu().detach()
        concept_labels_norm[cpt] = torch.matmul(image_feature_norm, concept_embedding).cpu().detach()
    Concept_Labels_org.append(concept_labels_org)
    Concept_Labels_norm.append(concept_labels_norm)

    if i % 10 == 0:
        print(f"Finish Extracting {i} Images!")

with open(f'concept_bank/{args.data_name}_concept_labels_original.pkl', 'wb') as file:
    pickle.dump(Concept_Labels_org, file)
with open(f'concept_bank/{args.data_name}_concept_labels_normalized.pkl', 'wb') as file:
    pickle.dump(Concept_Labels_norm, file)