import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()

dataset_path = args.dataset_path

while not dataset_path:
    print(f"Specify the dataset path")
    dataset_path = input("Enter path: ").strip()

splits = ['test', 'train', 'valid']

for split in splits:
    mask_dir = os.path.join(dataset_path, split, 'masks')
    output_label_dir = os.path.join(dataset_path, split, 'labels')

    if not os.path.exists(mask_dir):
        continue

    os.makedirs(output_label_dir, exist_ok=True)
    print(f"Processing {split} masks...")

    for mask_name in os.listdir(mask_dir):
        if not mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        # Threshold: Any pixel brighter than 10
        _, combined_binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        # Find the contours of the unified leaf shapes
        contours, _ = cv2.findContours(combined_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        base_name = os.path.splitext(mask_name)[0]
        if base_name.endswith('_mask'):
            base_name = base_name[:-5]

        txt_name = base_name + ".txt"

        with open(os.path.join(output_label_dir, txt_name), "w") as f:
            for contour in contours:
                if cv2.contourArea(contour) < 5:  # Filter out tiny pixel noise
                    continue

                polygon = []
                for point in contour:
                    x, y = point[0]
                    polygon.append(f"{x / w:.6f} {y / h:.6f}")

                # All detected leaves are assigned to Class 0
                class_id = 0

                if len(polygon) >= 3:
                    f.write(f"{class_id} " + " ".join(polygon) + "\n")

print("Conversion complete! All tones merged into a single leaf class.")