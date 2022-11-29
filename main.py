import matplotlib.pyplot as plt
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.loss import LabelSmoothingCrossEntropy

from tqdm import tqdm
import cv2
import time
import copy
import os
import imutils
import sys

cudnn.benchmark = True
plt.ion()

# Chaning to CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


'''
# Recommended data pre processing code generates cleaned folders

def crop_img(img):
    """
    Finds the extreme points on the image and creates a rectangular cropping
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    return new_img


if __name__ == "__main__":
    training = "dataset/Training"
    testing = "dataset/Testing"
    training_dir = os.listdir(training)
    testing_dir = os.listdir(testing)
    IMG_SIZE = 256

    for dir in training_dir:
        save_path = 'cleaned/Training/' + dir
        path = os.path.join(training, dir)
        image_dir = os.listdir(path)
        for img in image_dir:
            image = cv2.imread(os.path.join(path, img))
            new_img = crop_img(image)
            new_img = cv2.resize(new_img, (IMG_SIZE, IMG_SIZE))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + '/' + img, new_img)

    for dir in testing_dir:
        save_path = 'cleaned/Testing/' + dir
        path = os.path.join(testing, dir)
        image_dir = os.listdir(path)
        for img in image_dir:
            image = cv2.imread(os.path.join(path, img))
            new_img = crop_img(image)
            new_img = cv2.resize(new_img, (IMG_SIZE, IMG_SIZE))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + '/' + img, new_img)
'''


# Plotting data counts for train and test sets
def create_plots():
    notumor_train = len(os.listdir('cleaned/Training/notumor'))
    notumor_test = len(os.listdir('cleaned/Testing/notumor'))

    glioma_train = len(os.listdir('cleaned/Training/glioma'))
    glioma_test = len(os.listdir('cleaned/Testing/glioma'))

    meningioma_train = len(os.listdir('cleaned/Training/meningioma'))
    meningioma_test = len(os.listdir('cleaned/Testing/meningioma'))

    pituitary_train = len(os.listdir('cleaned/Training/pituitary'))
    pituitary_test = len(os.listdir('cleaned/Testing/pituitary'))

    classes = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    train_values = [notumor_train, glioma_train, meningioma_train, pituitary_train]
    test_values = [notumor_test, glioma_test, meningioma_test, pituitary_test]

    for i in range(4):
        test_proportion = (test_values[i]/(train_values[i] + test_values[i])) * 100
        print(f"{classes[i]}: {100-test_proportion : .2f} /{test_proportion : .2f}")

    X_axis = np.arange(len(classes))

    plt.title("Data Counts for Train and Test Sets")
    plt.xlabel("Classes")
    plt.ylabel("Counts")

    plt.bar(X_axis - 0.2, train_values, 0.4, label='Training')
    plt.bar(X_axis + 0.2, test_values, 0.4, label='Test')

    plt.xticks(X_axis, classes)

    plt.legend()
    plt.show()

    print("Train/Test proportions")


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    'Testing': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
}

data_dir = 'cleaned'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Training', 'Testing']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                          shuffle=True, num_workers=0)
           for x in ['Training', 'Testing']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Training', 'Testing']}
class_names = image_datasets['Training'].classes

# Displays some number of images from the set
'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(.01)  # pause a bit so that plots are updated
    plt.waitforbuttonpress() # plots disappear unless this is here
'''

'''
inputs, classes = next(iter(dataloaders['Training']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
'''

# Displays predictions for some number of images
'''
def visualize_model(model, num_images=16):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['Testing']):
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
'''


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'Training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Epoch Loss: {epoch_loss:.4f} Epoch Acc: {epoch_acc:.4f}')

            if phase == 'Training':
                writer.add_scalar('Training Epoch Loss', epoch_loss, epoch)
                writer.add_scalar('Training Epoch Accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('Testing Epoch Loss', epoch_loss, epoch)
                writer.add_scalar('Testing Epoch Accuracy', epoch_acc, epoch)


            # deep copy the model
            if phase == 'Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_noLRS(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Epoch Loss: {epoch_loss:.4f} Epoch Acc: {epoch_acc:.4f}')

            if phase == 'Training':
                writer.add_scalar('Training Epoch Loss', epoch_loss, epoch)
                writer.add_scalar('Training Epoch Accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('Testing Epoch Loss', epoch_loss, epoch)
                writer.add_scalar('Testing Epoch Accuracy', epoch_acc, epoch)


            # deep copy the model
            if phase == 'Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

writer = SummaryWriter('runs/exp1_Resnet18_SGDlr.001_momentum.8_smoothcross_noLRS')
images, labels = next(iter(dataloaders['Training']))

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_image('mri_images', img_grid)


def resnet_transfer():
    model_ft = models.resnet18(weights=True)
    num_ftrs = model_ft.fc.in_features

    # Change last layer to be linear
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    # Trying different cross entropy
    #criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()

    # Optimizer
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.88)
    #optimizer_ft = optim.NAdam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)

    # LR scheduler
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

    # No LR scheduler
    #model_ft = train_model_noLRS(model_ft, criterion, optimizer_ft, num_epochs=25)

    writer.add_graph(model_ft, images.to(device))

    # Basic prediction visualization
    # visualize_model(model_ft)

    #plt.ioff()
    #plt.show()


resnet_transfer()
writer.close()

