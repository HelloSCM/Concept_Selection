import os
import argparse
import random
import pickle
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataset.repre import RepresentationDataset


def get_args():
    parser = argparse.ArgumentParser(description='Concept Selection')
    # Core Setting
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'lad_a', 'lad_f', 'lad_v', 'lad_e', 'lad_h'])
    parser.add_argument('--cpt_path', type=str, default='./concept_bank/cifar10/cifar10_rough_selection_bar_0.pkl')
    parser.add_argument('--algorithm', type=str, default='cbm', choices=['lp', 'cbm', 'mask'])
    # Basic Setting
    parser.add_argument('--seed', type=int, default=42)
    # Training Setting
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--init_lr', type=float, default=0.005)
    parser.add_argument('--decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.8)
    # Tricks
    parser.add_argument('--cls_reg', type=float, default=0.0)
    parser.add_argument('--proj_reg', type=float, default=0.0)

    args = parser.parse_args()
    set_random_seed(args.seed)
    print("The Training Info: ")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_name, batch_size, cpt_path):
    trainset = RepresentationDataset(dataset=dataset_name, mode='train')
    testset = RepresentationDataset(dataset=dataset_name, mode='test')

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    with open(cpt_path, 'rb') as file:
        concept_bank = pickle.load(file)

    return trainloader, testloader, concept_bank


def record(args, model, metrics):
    result = pd.DataFrame.from_dict(metrics)
    best_epoch = np.argmin(np.array(metrics['Loss']))
    best_result = metrics['Acc'][best_epoch]
    print("========================================")
    print(f"Best Accuracy: {(100.0*best_result):.2f}%")
    
    if args.algorithm == 'lp':
        filename = f"Cpt_none_Norm_{args.norm}_Acc_{best_result:.4f}"
    else:
        filename = f"Cpt_{args.cpt_path.split('/')[-1].split('.')[0]}_Norm_{args.norm}_Acc_{best_result:.4f}"
    os.makedirs(f"./results/{args.dataset}/{args.algorithm}/{filename}", exist_ok=True)
    result_path = f"./results/{args.dataset}/{args.algorithm}/{filename}"
    
    args_info = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    result_info = (
        f"minimal_loss: {metrics['Loss'][best_epoch]:.4f}\n"
        f"best_accuracy: {(100.0*best_result):.2f}%"
    )
    with open(f"{result_path}/log_info.txt", 'w') as log_file:
        log_file.write(args_info + "\n" + result_info)
            
    result.to_csv(f"{result_path}/metrics.csv", index=False)
    torch.save(model, f"{result_path}/model.pt")
    
    print("Finish Saving Data.")