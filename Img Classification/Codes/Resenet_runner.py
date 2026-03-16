from PIL import Image
import torch
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes
model.load_state_dict(torch.load("D:/Img_Classification/best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
 transforms.Resize((224, 224)),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_path = "D:/img3.jpeg"
image = Image.open(img_path).convert("RGB")
image = transform(image)             # [3,224,224]
image = image.unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)                   # logits [1,2]
    probs = torch.softmax(outputs, dim=1)   # probabilities
    pred_class = torch.argmax(outputs, dim=1) 


class_names = ["Normal", "Pneumonia"]
print("Predicted class:", class_names[pred_class.item()])
print(f"Confidence: {probs[0, pred_class].item()*100:.2f}%")



# used to check confusion matrix
from sklearn.metrics import confusion_matrix

all_labels, all_preds = [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)