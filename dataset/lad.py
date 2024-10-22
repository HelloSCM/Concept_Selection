import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LAD(Dataset):
    def __init__(self, root, data_type, transform, mode='train', concept_bottleneck=True):
        with open(f"{root}attributes.txt", 'r') as f:
            train_lines = f.readlines()
        with open(f"{root}images.txt", 'r') as f:
            all_lines = f.readlines()
        with open(f"{root}label_list.txt", 'r') as f:
            class_lines = f.readlines()
        
        classes = []
        for l in class_lines:
            if data_type in l.split(', ')[0]:
                classes.append(l.split(', ')[1])
        self.classes = sorted(classes)

        concept_interval = {'A': [0, 123], 'F': [123, 181], 'V': [181, 262], 'E': [262, 337], 'H': [337, 359]}
        train_img_paths, train_labels, concepts = [], [], []
        for l in train_lines:
            if data_type in l.split(', ')[0]:
                train_img_paths.append(root + l.split(', ')[1])
                train_labels.append(int(l.split(', ')[0][-2:])-1)
                concepts.append(np.array([int(c) for c in l.split(', ')[2].split()[1:-1]][concept_interval[data_type][0]:concept_interval[data_type][1]]))

        test_img_paths, test_labels = [], []
        for l in all_lines:
            if data_type in l.split(', ')[1]:
                if l.split(', ')[6][:-1] not in train_img_paths:
                    test_img_paths.append(root + l.split(', ')[6][:-1])
                    test_labels.append(int(l.split(', ')[1][-2:])-1)
        
        if mode == 'train':
            self.img_paths = train_img_paths
            self.targets = train_labels
            self.concepts = concepts
        elif mode == 'test':
            self.img_paths = test_img_paths
            self.targets = test_labels
        
        self.transform = transform
        self.mode = mode
        self.concept_bottleneck = concept_bottleneck


    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, idx):
        if self.mode == 'train' and self.concept_bottleneck == True:
            label = self.targets[idx]
            cpt = self.concepts[idx]
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label, cpt

        else:
            label = self.targets[idx]
            img_path = self.img_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label