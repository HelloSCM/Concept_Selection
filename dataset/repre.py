import pickle
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from dataset.lad import LAD


class RepresentationDataset(Dataset):
    def __init__(self, dataset, mode):
        if mode == 'train':
            self.is_train = True
        elif mode == 'test':
            self.is_train = False

        if dataset == 'cifar10':
            img_dataset = CIFAR10(root='/data/.../', train=self.is_train, download=False, transform='None')
        elif dataset == 'cifar100':
            img_dataset = CIFAR100(root='/data/.../', train=self.is_train, download=False, transform='None')
        elif dataset[:3] == 'lad':
            img_dataset = LAD(root='/data/.../', data_type=dataset[-1].upper(), transform='None', mode=mode, concept_bottleneck=False)
        else:
            raise ValueError('Wrong Dataset!')

        self.targets = img_dataset.targets
        with open(f'/home/scm22/ConceptSelection/concept_bank/{dataset}/{dataset}_image_representation_{mode}.pkl', 'rb') as file:
            repres = pickle.load(file)
        self.repres = torch.from_numpy(repres)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.repres[idx], self.targets[idx]