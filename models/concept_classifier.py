import torch
import torch.nn as nn


class ConceptClassifier(nn.Module):
    def __init__(self, class_num, concept_embeds, cls_reg):
        super(ConceptClassifier, self).__init__()
        self.classifier = nn.Linear(concept_embeds.shape[0], class_num)
        self.fea_cpts = concept_embeds
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
        cpts = torch.matmul(fea_img, self.fea_cpts.T).float()
        cpts_normalized = (cpts - torch.mean(cpts, dim=1, keepdim=True)) / torch.std(cpts, dim=1, keepdim=True)
        out = self.classifier(cpts_normalized)
        
        return out