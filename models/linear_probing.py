import torch
import torch.nn as nn


class LinearProbing(nn.Module):
    def __init__(self, class_num, fea_dim, cls_reg):
        super(LinearProbing, self).__init__()
        self.classifier = nn.Linear(fea_dim, class_num)
        self.cls_reg = cls_reg


    def update(self, feas, labels, criterion, optimizer):
        preds = self.predict(feas)
        loss = criterion(preds, labels)

        l1_reg = 0.0
        for param in self.classifier.parameters():
            l1_reg += torch.norm(param, 1)
        loss += self.cls_reg * l1_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    

    def predict(self, fea_img):
        out = self.classifier(fea_img.float())
        
        return out