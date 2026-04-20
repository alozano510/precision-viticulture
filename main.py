import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Set GPU for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load ResNet model
model = models.resnet50(weights=None)

# Change final layer to be a binary classifier
num_classes = 2  # healthy / unhealthy
class_names = ['no saludable', 'saludable']
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Load weights from fine-tuning
model.load_state_dict(torch.load('model\\binary_classifier_v1.pt', map_location=device))
print("Model weights loaded\n")
model.to(device)
model.eval()

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def list_available_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


camera = 1

source = cv2.VideoCapture(camera)

if not source.isOpened():
    raise ValueError("Error: Could not open camera")

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        print("Could not read frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        _, pred = torch.max(output, 1)
        label = class_names[pred.item()]
        confidence = probs[pred.item()].item() * 100

    # Draw prediction on frame
    color = (0, 255, 0) if label == 'saludable' else (0, 0, 255)
    cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2)

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)