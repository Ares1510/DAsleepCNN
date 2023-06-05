import math
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, AUROC, F1Score
from models.functions import ReverseLayerF


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = x.unsqueeze(1).float()
        x = self.features(x)
        return x
    
class LabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12, 48),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(48, 1)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class CNN3L_DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.label_classifier = LabelClassifier()
        self.domain_classifier = DomainClassifier()

    def forward(self, x, alpha=0.1):
        x = self.features(x)
        reverse_x = ReverseLayerF.apply(x, alpha)
        label_output = self.label_classifier(x)
        domain_output = self.domain_classifier(reverse_x)
        return label_output, domain_output



#class to implement normal CNN3L with lightning
class CNN3L_Lightning(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.cnn = FeatureExtractor()
        self.lc = LabelClassifier()
        
        self.criterion = nn.BCEWithLogitsLoss()

        metrics = MetricCollection([Accuracy(task='binary'), 
                                               AUROC(task='binary'), 
                                               F1Score(task='binary', average='macro')])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.target_metrics = metrics.clone(prefix='target_')
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        output = self.lc(self.cnn(X)).squeeze()
        loss = self.criterion(output, y.float())
        #compute metrics and log them
        self.train_metrics(output, y, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        output = self.lc(self.cnn(X)).squeeze()
        loss = self.criterion(output, y.float())
        self.val_metrics(output, y, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        output = self.lc(self.cnn(X)).squeeze()
        loss = self.criterion(output, y.float())
        self.log('test_loss', loss, on_step=True, logger=True)

        if dataloader_idx == 0:
            self.test_metrics(output, y, on_epoch=True)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        elif dataloader_idx == 1:
            self.target_metrics(output, y, on_epoch=True)
            self.log_dict(self.target_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    

#class to implement DANN with lightning
class DANN_Lightning(pl.LightningModule):
    def __init__(self, source_loader, target_loader, lr=0.001):
        super().__init__()
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.lr = lr
        self.alpha = 10
        self.beta = 0.75
        # self.save_hyperparameters()

        self.model = CNN3L_DANN()
        self.criterion = nn.BCEWithLogitsLoss()

        metrics = MetricCollection([Accuracy(task='binary'), 
                                               AUROC(task='binary'), 
                                               F1Score(task='binary', average='macro')])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.target_metrics = metrics.clone(prefix='target_')
    
    def training_step(self, batch, batch_idx):
        source_data, source_labels = batch['source']
        target_data, _ = batch['target']

        epoch = self.current_epoch
        p = (batch_idx + epoch * len(self.source_loader)) / (self.trainer.max_epochs * len(self.source_loader))
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_domain_labels = torch.ones(source_data.shape[0]).to(self.device)

        label_output, domain_output = self.model(source_data, alpha=alpha)
        label_loss = self.criterion(label_output.squeeze(), source_labels.float())
        source_domain_loss = self.criterion(domain_output.squeeze(), source_domain_labels.float())

        target_domain_labels = torch.zeros(target_data.shape[0]).to(self.device)
        _, domain_output = self.model(target_data, alpha=alpha)
        target_domain_loss = self.criterion(domain_output.squeeze(), target_domain_labels.float())

        loss = label_loss + source_domain_loss + target_domain_loss

        self.train_metrics(label_output.squeeze(), source_labels, on_epoch=True, batch_size=source_data.shape[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=source_data.shape[0])
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, batch_size=source_data.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        label_output, _ = self.model(X)
        loss = self.criterion(label_output.squeeze(), y.float())
        self.val_metrics(label_output.squeeze(), y, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        label_output, _ = self.model(X)
        loss = self.criterion(label_output.squeeze(), y.float())
        self.log('test_loss', loss, on_step=True, logger=True)

        if dataloader_idx == 0:
            self.test_metrics(label_output.squeeze(), y, on_epoch=True)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        elif dataloader_idx == 1:
            self.target_metrics(label_output.squeeze(), y, on_epoch=True)
            self.log_dict(self.target_metrics, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def train_dataloader(self):
        return {'source': self.source_loader, 'target': self.target_loader}
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
