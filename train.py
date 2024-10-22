import torch
import torch.nn as nn
import torch.optim as optim

from models.linear_probing import LinearProbing
from models.concept_classifier import ConceptClassifier
from models.mask_concept_classifier import MaskConceptClassifier
from utils import *


def train(epochs, trainloader, testloader, model, criterion, optimizer, scheduler):
    metrics = {'Loss':[], 'Acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_id, (repres, labels) in enumerate(trainloader):
            repres, labels = repres.cuda(), labels.cuda()
            batch_result = model.update(repres, labels, criterion, optimizer)
            running_loss += batch_result
        scheduler.step()

        loss = running_loss / (batch_id + 1)
        print(f"==================== Epoch [{epoch+1}/{epochs}] ====================")
        print(f"Train - Loss: {loss:.4f}")
        acc = eval(model, testloader)
        
        metrics['Loss'].append(loss)
        metrics['Acc'].append(acc)

    return model, metrics


def eval(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for repres, labels in testloader:
            repres, labels = repres.cuda(), labels.cuda()
            outputs = model.predict(repres)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()        
    acc = correct / total
    print(f"Test - Accuracy: {(100.0*acc):.2f}%")
    
    return acc


def main():
    args = get_args()
    trainloader, testloader, concept_bank = load_data(args.dataset, args.shot, args.batch_size, args.cpt_path)
    cls_num = len(torch.unique(torch.tensor(trainloader.dataset.targets)))
    cpt_names = list(concept_bank.keys())
    cpt_embeds = torch.stack(list(concept_bank.values())).cuda()

    if args.algorithm == 'lp':
        model = LinearProbing(cls_num, cpt_embeds.shape[1], args.cls_reg)
    elif args.algorithm == 'cbm':
        model = ConceptClassifier(cls_num, cpt_embeds, args.cls_reg)
    elif args.algorithm == 'mask':
        model = MaskConceptClassifier(cls_num, cpt_embeds, args.cls_reg)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    model, metrics = train(args.epochs, trainloader, testloader, model, criterion, optimizer, scheduler)
    record(args, model, metrics)


if __name__ == '__main__':
    main()