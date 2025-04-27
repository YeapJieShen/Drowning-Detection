
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, classification_report
import os
from collections import defaultdict
from tqdm import tqdm

class BalancedDataset(Dataset):
    def __init__(self, data_dir, original_dataset, aug_transform=None, enhance_transform=None):
        self.original_dataset = original_dataset
        self.data_dir = data_dir
        self.aug_transform = aug_transform
        self.enhance_transform = enhance_transform

        self.imgs_by_class = self._group_by_class()
        self.max_class_count = max(len(imgs) for imgs in self.imgs_by_class.values())

        # Flatten balanced dataset
        self.balanced_data = []
        for label, imgs in self.imgs_by_class.items():
            for i in range(self.max_class_count):
                self.balanced_data.append((imgs[i % len(imgs)], label, i >= len(imgs)))  # (img_path, label, is_augmented)

    def _group_by_class(self):
        class_dict = {label: [] for label in self.original_dataset.class_to_idx.values()}
        for path, label in self.original_dataset.imgs:
            class_dict[label].append(path)
        return class_dict

    def __len__(self):
        return len(self.balanced_data)

    def __getitem__(self, idx):
        img_path, label, is_augmented = self.balanced_data[idx]
        img = Image.open(img_path).convert("RGB")

        if is_augmented and self.aug_transform:
            img = self.aug_transform(img)

        if self.enhance_transform:
            img = self.enhance_transform(img)

        return img, label

class TorchClassifierTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        if 'device' in config:
            self.device = torch.device(config['device'])
        else:
            self.device = next(model.parameters()).device

        self.imbalance = config.get('imbalance', True)
        self.val_test_ratio = config.get('val_test_ratio', 0.5)
        self.input_size = self.config.get('input_size', 128)

        self.optimizer = None
        self.criterion = None

        self.train_loader = None
        self.val_loader = None

    def _get_loader(self, dataset, batch_size=32, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def _get_optimizer(self, model, name, lr=1e-3, momentum=0.9, betas=(0.9, 0.999), weight_decay=0, alpha=0.99):
        if name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif name == 'RMSprop':
            return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _get_criterion(self, name='CrossEntropyLoss'):
        if name == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {name}")

    @staticmethod
    def stratified_subset(imagefolder, fraction=1.0):
        grouped = defaultdict(list)

        for img_path, label in imagefolder.imgs:
            grouped[label].append((img_path, label))

        subset = []
        for label, samples in grouped.items():
            k = max(1, int(len(samples) * fraction))
            subset.extend(samples[:k])  # stratified slicing

        imagefolder.imgs = subset
        imagefolder.samples = subset
        return imagefolder

    def _prepare_train(self):
        data_path = self.config['data_path']

        self.criterion = self._get_criterion(
            self.config.get('loss', 'CrossEntropyLoss'))
        self.optimizer = self._get_optimizer(
            self.model,
            name=self.config.get('optimizer', 'Adam'),
            lr=self.config.get('lr', 1e-3),
            momentum=self.config.get('momentum', 0.9),
            betas=self.config.get('betas', (0.9, 0.999)),
            weight_decay=self.config.get('weight_decay', 0),
            alpha=self.config.get('alpha', 0.99)
        )

        val_dir = os.path.join(data_path, 'val')

        val_dataset = ImageFolder(val_dir, transform=self.config['val_transform'])
        self.class_labels = val_dataset.classes
        self.class_to_idx = val_dataset.class_to_idx

        val_dataset = self.stratified_subset(val_dataset, fraction=self.config['val_test_ratio'])

        train_dir = os.path.join(data_path, 'train')

        if self.imbalance:
            # Do some augmentation to make training classes balance
            original_training_dataset = ImageFolder(train_dir, transform=None)
            original_training_dataset = self.stratified_subset(original_training_dataset, fraction=self.config.get('fraction', 1.0))

            train_dataset = BalancedDataset(
                train_dir, original_training_dataset, aug_transform=self.config.get('aug_transform', None), enhance_transform=self.config['enhance_transform'])
        else:
            train_dataset = ImageFolder(
                train_dir, transform=self.config['enhance_transform'])
            train_dataset = self.stratified_subset(train_dataset, fraction=self.config.get('fraction', 1.0))
            

        val_loader = self._get_loader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
        )

        train_loader = self._get_loader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
        )

        self.val_loader = val_loader
        self.train_loader = train_loader

    def train(self):
        self._prepare_train()

        train_logs = []
        val_logs = []

        epochs = self.config.get('epochs', 50)
        train_loader = self.train_loader
        val_loader = self.val_loader
        optimizer = self.optimizer
        criterion = self.criterion

        for epoch in range(epochs):
            if epoch == int(epochs * 0.5):
                self.optimizer.param_groups[0]['lr'] *= 0.1

            train_metrics = self._train_epoch(
                epoch, epochs, train_loader, optimizer, criterion)
            train_logs.append(train_metrics)

            val_metrics = self._validate_epoch(
                epoch, epochs, val_loader, criterion)
            val_logs.append(val_metrics)

        return train_logs, val_logs

    def _train_epoch(self, epoch, epochs, dataloader, optimizer, criterion):
        self.model.train()

        running_loss = 0.0

        correct_preds = 0
        total_preds = 10
        all_labels = []
        all_preds = []

        print(
            ("%11s" * 9)
            %
            (
                "Epoch",
                "Loss",
                "Accuracy",
                "Macro(P",
                "R",
                "F1)",
                "Weighted(P",
                "R",
                "F1)"
            )
        )

        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=
                ("%11s" + "%11.4g" * 8)
                % (
                    f"{epoch + 1}/{epochs}",
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.
                )
            )

        for batch_idx, (inputs, targets) in progress:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if batch_idx % 2 == 0:
                macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='macro', zero_division=0
                )
                weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )

                progress.set_description(
                    ("%11s" + "%11.4g" * 8)
                    % (
                        f"{epoch + 1}/{epochs}",
                        running_loss / (batch_idx + 1),
                        100 * correct_preds / total_preds,
                        macro_precision,
                        macro_recall,
                        macro_f1,
                        weighted_precision,
                        weighted_recall,
                        weighted_f1
                    )
                )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        return {
            'loss': running_loss / len(dataloader),
            'accuracy': 100 * correct_preds / total_preds,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'weighted_precision': weighted_precision,
            'macro_recall': macro_recall,
            'weighted_recall': weighted_recall
        }

    def _validate_epoch(self, epoch, epochs, dataloader, criterion):
        self.model.eval()

        running_loss = 0.0

        all_labels = []
        all_preds = []

        print(
            ("%11s" * 2)
            %
            (
                "",
                "Loss"
            )
        )

        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=
                ("%11s" + "%11.4g")
                % (
                    "",
                    0.,
                )
            )

        with torch.no_grad():
            for batch_idx, (inputs, targets) in progress:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                progress.set_description(
                    ("%11s" + "%11.4g")
                    % (
                        "",
                        running_loss / (batch_idx + 1),
                    )
                )

        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=self.class_labels,
                digits=4,
                zero_division=0
            )
        )

        return {
            'loss': running_loss / len(dataloader),
            **classification_report(
                all_labels,
                all_preds,
                target_names=self.class_labels,
                digits=4,
                zero_division=0,
                output_dict=True
            )
        }