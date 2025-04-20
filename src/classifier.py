import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall

class ImageClassificationBase(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.val_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.val_recall_per_class = MulticlassRecall(num_classes=num_classes, average=None)

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.train_acc_metric(out, labels)
        return {'train_loss': loss, 'train_acc': acc}

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.val_acc_metric(out, labels)
        accs = self.val_acc_per_class(out, labels)
        recalls = self.val_recall_per_class(out, labels)
        return {
            'val_loss': loss.detach(),
            'val_acc': acc.detach(),
            'val_recalls': recalls.detach(),
            'val_accs': accs.detach()
        }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accs = [x['val_acc'] for x in outputs]
        batch_recalls = [x['val_recalls'] for x in outputs]
        batch_accs_cls = [x['val_accs'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()
        epoch_acc = torch.stack(batch_accs).mean()
        recalls_per_class = torch.stack(batch_recalls).mean(dim=0).tolist()
        accs_per_class = torch.stack(batch_accs_cls).mean(dim=0).tolist()

        return {
            'val_loss': epoch_loss.item(),
            'val_acc': epoch_acc.item(),
            'val_recalls': recalls_per_class,
            'val_accs_cls': accs_per_class
        }

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], "
              f"train_loss: {result['train_loss']:.4f}, train_acc: {result['train_acc']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        
        print("Per-class recall:")
        for class_id, recall in enumerate(result['val_recalls']):
            print(f"  Class {class_id}: {recall:.4f}")
        
        print("Per-class accuracy:")
        for class_id, acc in enumerate(result['val_accs_cls']):
            print(f"  Class {class_id}: {acc:.4f}")

class SimpleCNN(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb, return_probs=False):
        x = self.network(xb)
        x = self.fc_layers(x)
        return F.softmax(x, dim=1) if return_probs else x
