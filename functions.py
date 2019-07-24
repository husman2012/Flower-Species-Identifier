import time
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import torch
from torchvision import transforms, datasets, models
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import json
from PIL import Image
import utils
import os

def cat_to_name(train_data, category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    for key in cat_to_name.keys():
        new_label = cat_to_name[key]
        train_data.class_to_idx[new_label] = train_data.class_to_idx.pop(key)
        
    return train_data.class_to_idx
        
def save_model(model, train_data, hidden_units, features, filepath = ''):
    model.class_to_idx = train_data.class_to_idx
    state_dict = model.state_dict()
    model.hidden_units = hidden_units
    model.input_nodes = features
    torch.save(model,'classifier_checkpoint.pth')
    torch.save(state_dict,'classifier_checkpoint_state_dict.pth')
    torch.save(model.class_to_idx, filepath + 'classifier_checkpoint_classes.pth')
    torch.save(model.hidden_units, filepath + 'classifier_checkpoint_hidden.pth')
    torch.save(model.input_nodes, filepath + 'classifier_checkpoint_input.pth')


def load_model(checkpoint, gpu, filepath = ''):
    checkpoint = torch.load(checkpoint)
    checkpoint.hidden_units = torch.load(filepath +'classifier_checkpoint_hidden.pth')
    model, criterion, optimizer, hidden_units, input_nodes = build_model(gpu = gpu, hidden_units = checkpoint.hidden_units)
    model.load_state_dict(torch.load('classifier_checkpoint_state_dict.pth'))
    model.class_to_idx = torch.load(filepath + 'classifier_checkpoint_classes.pth')
    return model

def predict(image, model, top_k):
    image = process_image(image)
    image = torch.Tensor(image)
    image.unsqueeze_(0)
    model = model.to('cuda')
    image = image.to('cuda')
    

    with torch.no_grad():
        output = model(image)
    ps = torch.exp(output).topk(top_k)
    probs = ps[0].tolist()[0]
    classes = ps[1].tolist()[0]
    
    name_list = []
    for number in classes:
        for key in model.class_to_idx.keys():
            if model.class_to_idx[key] == number:
                name_list.append(key)
                
                
    return probs, name_list


def process_image(image_dir):
    image = Image.open(image_dir)
    if image.width > image.height:
        factor = image.width/image.height
        new_width = round(factor*256)
        image = image.resize(size = (int(round(new_width)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size = (256, int(round(factor*256, 0))))
    
    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
    # Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def print_results(probs, name_list, top_k):
    print('\nProbabilities' + '   |   ' + 'Name of Flower')
    print('-------------------------------------------------')

    for item in range(top_k):
        print(f'{probs[item]:.2f}            |             {name_list[item]}')
        
def build_model(gpu, arch = 'resnet18', learning_rate = 0.001, hidden_units = 250, features = 0, train_dir = 'flowers/train/'):
    if gpu == True:
        device = torch.device ('cuda')
    else:
        device = torch.device('cpu')
    print(type(arch))
    model, features = utils.get_model(arch)
    
    for param in model.parameters():
        param.requires_grad = False
    n_output = sum(os.path.isdir(os.path.join(train_dir, i)) for i in os.listdir(train_dir))
    print(n_output)
    if arch in ['resnet101', 'resnet152', 'resnet18','resnet34','resnet50','inception_v3']:
        model.fc = nn.Sequential(nn.Linear(features, int(hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units), n_output),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr = float(learning_rate))
        model.to(device)
        
    elif arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        model.classifier = nn.Sequential(nn.Linear(features, int(hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units), n_output),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
        model.to(device)
        
    elif arch in ['alexnet']:
        model.classifier = nn.Sequential(nn.Linear(9216, int(hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units), n_output),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
        model.to(device)    
        
    elif arch in ['squeezenet1_0','squeezenet1_1']:
        model.classifier[1] = nn.Conv2d(512, 102, kernel_size=(1,1), stride=(1,1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
        model.to(device)    
        
    else:
        model.classifier = nn.Sequential(nn.Linear(25088, int(hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(hidden_units), n_output),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
        model.to(device)
        
    return model, device, optimizer, criterion, features


def create_loader(data_dir):
    transform = transforms.Compose([transforms.Resize(255), 
                                    transforms.CenterCrop(224),  
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    data = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)
    return loader, data
    
def train_model(model, epochs, trainloader, validloader, device, optimizer, criterion):
    epochs = int(epochs)
    steps = 0
    running_loss = 0
    print_every = 5
    start = time.time()
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs);
            loss= criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)                  
                        test_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    total_time = (time.time() - start)//60
    total_time_s = (time.time() - start)%60
    print(str(total_time) + ' minutes ' + str(total_time_s) + ' seconds')
    return model
    
    
def test_model(model, testloader, device):
    test_losses = []
    test_loss = 0
    accuracy = 0    
    model.eval()
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            loss = test_loss.item()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
                
    test_losses.append(test_loss/len(testloader))

    print("Test Set Loss: {:.3f}.. ".format(loss/len(testloader)),
              "Test Set Accuracy: {:.3f}".format(accuracy/len(testloader)))
    