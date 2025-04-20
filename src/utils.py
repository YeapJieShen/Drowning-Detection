from PIL import Image
import torch
from torchvision import transforms
from data_loader import CustomTransformation

def predict_image(image_path, model, image_size=(128, 128)):
    classes = ['Swimming', 'Treading Water', 'Drowning']
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
            transforms.Resize(image_size),
            CustomTransformation(),
            transforms.ToTensor(),
        ])
    
    img_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor, return_probs=True)
        prob, pred = torch.max(outputs, dim=1)

    return classes[pred.item()], prob.item()
