from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from data_loader import CustomTransformation

class Tester:
    def __init__(self, model, test_dataset_path, batch_size=32, image_size=(128, 128)):
        self.model = model
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.transformations = transforms.Compose([
            transforms.Resize(image_size), 
            CustomTransformation, 
            transforms.ToTensor()])
        
        # Load the test dataset and create DataLoader
        self.test_dataset = datasets.ImageFolder(self.test_dataset_path, transform=self.transformations)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def evaluate(self):
        self.model.eval()
        # Assuming evaluate function returns a dictionary with 'val_loss' and 'val_acc'
        outputs = [self.model.validation_step(batch) for batch in self.test_loader]
        return self.model.validation_epoch_end(outputs)
    
    def test(self):
        # Start timer for evaluation
        start_time = time.time()

        # Evaluate the model
        test_result = self.evaluate()

        # End timer
        end_time = time.time()

        # Calculate time taken
        time_taken = end_time - start_time

        # Display results
        print(f"Test Loss: {test_result['val_loss']:.4f}")
        print(f"Test Accuracy: {test_result['val_acc']:.4f}")

        # Display per-class recalls
        print("Per-class Recall:")
        for idx, recall in enumerate(test_result['val_recalls']):
            print(f"  Class {idx}: {recall:.4f}")

        # Display per-class accuracies
        print("Per-class Accuracy:")
        for idx, acc in enumerate(test_result['val_accs_cls']):
            print(f"  Class {idx}: {acc:.4f}")

        print(f"Time Taken for Testing: {time_taken:.2f} seconds")
