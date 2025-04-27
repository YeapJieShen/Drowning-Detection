import torch
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
import os
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import yaml
from .models import CNNClassifier
from .trainer import TorchClassifierTrainer

MODEL_REGISTRY = {
    'CNNClassifier': CNNClassifier
}


class TorchClassifier(torch.nn.Module):
    def __init__(self, model, model_path=None, config=None, device='auto', verbose=False):
        super().__init__()

        config = self._load_config(config) if config else None

        self.model_cls = self._get_model_cls(model)
        self.model_path = model_path
        self.config = config

        self.verbose = verbose
        self.device = torch.device(
            device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.trainer = None

        if model_path:
            self.load(model_path)
        elif config:
            self._new()

    def _get_model_cls(self, model_name):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name: {model_name}")
        return MODEL_REGISTRY[model_name]

    def _new(self):
        if isinstance(self.config, dict):
            self.model = self.model_cls(**self.config).to(self.device)
        else:
            raise ValueError("Configuration must be a dictionary!")

        if self.verbose:
            print(f"New model created: {self.model}")

    def __call__(self, img, transform=None, prob=False):
        return self.predict(img, transform, prob)

    def _load_config(self, config):
        if isinstance(config, dict):
            return config

        config_path = Path(config)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file {config_path} not found!")

        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.suffix in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {config_path.suffix}")

    def load(self, weights):
        if Path(weights).suffix != '.pt':
            raise ValueError(f"Model file must be a .pt file, got {weights}")

        checkpoint = torch.load(weights, map_location='cpu')

        config = checkpoint.get("config")

        if config is None:
            raise ValueError("Missing model configuration in checkpoint!")

        self.model = self.model_cls(**config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        if self.verbose:
            print(f"Model loaded from {weights}")
            print(f"Model configuration: {config}")

    def train(self, **kwargs):
        self.trainer = TorchClassifierTrainer(self.model, kwargs)
        self.training_logs = self.trainer.train()

        self.class_to_idx = self.trainer.class_to_idx
        return self.training_logs

    @staticmethod
    def stratified_subset(imagefolder, fraction=1.0, head=True):
        grouped = defaultdict(list)

        for img_path, label in imagefolder.imgs:
            grouped[label].append((img_path, label))

        subset = []
        for label, samples in grouped.items():
            k = max(1, int(len(samples) * fraction))
            if head:
                subset.extend(samples[:k])
            else:
                subset.extend(samples[-k:])

        imagefolder.imgs = subset
        imagefolder.samples = subset
        return imagefolder
    
    def validate(self, data_path, transform, val_test_ratio):
        val_dir = os.path.join(data_path, 'val')

        val_dataset = ImageFolder(val_dir, transform=transform)
        class_labels = val_dataset.classes

        val_dataset = self.stratified_subset(
            val_dataset, fraction=val_test_ratio, head=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )

        self.model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=class_labels,
                digits=4,
                zero_division=0
            )
        )

        return {
            **classification_report(
                all_labels,
                all_preds,
                target_names=class_labels,
                digits=4,
                zero_division=0,
                output_dict=True
            )
        }

    def predict(self, img, transform=None, prob=False):
        if transform:
            img = transform(img)

        with torch.no_grad():
            self.model.eval()

            img = img.unsqueeze(0).to(self.device)
            output = self.model(img, prob=prob)

        return output.squeeze(0)

    def save(self, filename):
        if not filename.endswith('.pt'):
            raise ValueError("Model file must be a .pt file!")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, filename)

        if self.verbose:
            print(f"Model saved to {filename}")
