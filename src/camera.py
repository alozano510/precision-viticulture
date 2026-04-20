import cv2
from PIL import Image
from sympy.printing.pytorch import torch


def list_available_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

camera = 2

source = cv2.VideoCapture(camera)

if not source.isOpened():
    raise ValueError("Error: Could not open camera")

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def leaf_health_classifier(model, transform, device):
    raise NotImplementedError

def camera_monitoring():
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
            probabilities = torch.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)
            label = class_names[pred.item()]
            confidence = probabilites[pred.item()].item() * 100

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
