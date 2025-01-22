import os
import sys
import torch



from model import VLR
from utils import CLIP_pipeline, classification_metrics, devices


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


proportion=[0.7,0.15,0.15]
criterion = nn.CrossEntropyLoss()
learning_rate = 2e-5
weight_decay = 0.01
num_epochs = 35
batch_size = 16


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'data_loader'))

# import VLM_pheme_dataloader
# train_loader,test_loader = VLM_pheme_dataloader.load_data_pheme(batch_size)



import VLM_MR2_en_dataloader
train_loader = VLM_MR2_en_dataloader.load_train_MR2(batch_size)
test_loader = VLM_MR2_en_dataloader.load_test_MR2(batch_size)





class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for x1, x2, x3, x4, labels in self.train_loader:
            
            x1, x2, x3, x4= CLIP_pipeline(x1, x2, x3, x4)
            x1, x2, x3, x4, labels = x1.to(self.device), x2.to(self.device), x3.to(self.device), x4.to(self.device),labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x1, x2, x3, x4)
            output_1 = outputs["label_1"]
            output_2 = outputs["label_2"]
            output_3 = outputs["label_3"]
            
            loss_1 = self.criterion(output_1, labels)
            loss_2 = self.criterion(output_2, labels)
            loss_3 = self.criterion(output_3, labels)
            loss = loss_1*proportion[0]+ loss_2*proportion[1]+loss_3*proportion[2]
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * x1.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss


    def test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        

        all_predicted = []
        all_labels = []
        
        with torch.no_grad():
            for x1, x2, x3, x4, labels in self.test_loader:
                
                x1, x2, x3, x4= CLIP_pipeline(x1, x2, x3, x4)
                x1, x2, x3, x4, labels = x1.to(self.device), x2.to(self.device), x3.to(self.device), x4.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x1, x2, x3, x4)["label_1"]
                
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * x1.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predicted.extend(predicted.cpu())
                all_labels.extend(labels.cpu())
                
        epoch_loss = running_loss / len(self.test_loader.dataset)
        accuracy = correct / total
        
        classification_metrics(torch.tensor(all_predicted), torch.tensor(all_labels))
        
        return epoch_loss, accuracy

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.test()
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')




if __name__ == '__main__':

    model = VLR(dim=768).to(devices)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Trainer
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device=devices)
    print("Training...")
    trainer.fit(num_epochs)
    



