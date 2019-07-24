from functions import *
import argparse

parser = argparse.ArgumentParser(description = 'train')

parser.add_argument('data_dir', type = str)
parser.add_argument('--save_dir', default = '')
parser.add_argument('--arch', default = 'resnet18')
parser.add_argument('--learning_rate', default = 0.001, type = int)
parser.add_argument('--hidden_units', default = 250)
parser.add_argument('--epochs', default = 1)
parser.add_argument('--gpu', action='store_true', default = False)
args = parser.parse_args()

trainloader, traindata = create_loader(args.data_dir + '/train')
testloader, testdata = create_loader(args.data_dir + '/test')
validloader, validdata = create_loader(args.data_dir + '/valid')
model, device, optimizer, criterion, features = build_model(args.gpu, args.arch, args.learning_rate, args.hidden_units)
trained_model = train_model(model, args.epochs, trainloader, validloader, device, optimizer, criterion)
test_model(trained_model, testloader, device)
save_model(trained_model, traindata, args.hidden_units, features)