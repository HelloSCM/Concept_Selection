import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root, transform, mode='train'):
        with open(f'{root}class_names.txt', 'r') as f:
            class_lines = f.readlines()
        self.classes = []
        for l in class_lines:
            name = l.split(': ')[1].split(', ')[0]
            if name[-1] == '\n':
                name = name[:-1]
            self.classes.append(name)

        self.img_paths = []
        self.targets = []

        if mode == 'train':
            class_path = os.path.join(root, 'train')
        elif mode == 'test':
            class_path = os.path.join(root, 'val')
        for i, c in enumerate(sorted(os.listdir(class_path))):
            image_path = os.path.join(class_path, c)
            cls_shot = 0
            for img_name in os.listdir(image_path):
                if img_name[-3:] == 'txt':
                    continue
                img_path = os.path.join(image_path, img_name)
                self.img_paths.append(img_path)
                self.targets.append(i)
                cls_shot += 1

        self.transform = transform

    
    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, idx):
        label = self.targets[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label