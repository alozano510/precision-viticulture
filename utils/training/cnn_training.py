import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.optim import lr_scheduler
from tempfile import TemporaryDirectory

# Data augmentation and normalization for training
# Data normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # this is supposedly important for plant datasets (todo: research this)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}

remap = {0: 0, 1: 0, 2: 1, 3: 0} # 0 = no saludable, 1 = saludable

def remap_label(label):
    return remap[label]

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with (TemporaryDirectory() as tempdir):
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set models to training mode
                else:
                    model.eval()   # Set models to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Counters for Precision and Recall
                TP = 0
                TN = 0
                FP = 0
                FN = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    TP += torch.sum((preds == 1) & (labels == 1)).item()
                    TN += torch.sum((preds == 0) & (labels == 0)).item()
                    FP += torch.sum((preds == 1) & (labels == 0)).item()
                    FN += torch.sum((preds == 0) & (labels == 1)).item()

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                epoch_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Prec: {epoch_prec:.4f} Recall: {epoch_recall:.4f}')

                # deep copy the models
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best models weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__=='__main__':
    # Set GPU for processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = "/data"
    image_datasets = {x: datasets.ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transforms[x],
            target_transform=remap_label)
        for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                   shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['no saludable', 'saludable']

    # Load ResNet models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Change final layer to be a binary classifier
    num_classes = 2  # healthy / unhealthy
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Move models to GPU
    model = model.to(device)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # for param in models.layer3.parameters():
    #     param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimization algorithm
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    # Decay Learning Rate by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    tuned_model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=10)

    visualize_model(tuned_model)

    torch.save(tuned_model.state_dict(), "/models/cnn_binary_classifier_v1.pt")
    print("Model saved!")