# Flower-Species-Identifier
This is a repository for a Udacity Project that utilizes PyTorch to create a command-line application capable of implementing the training of multiple neural networks, and running predictions on Flower images in PIL format. This Readme will be broken up into two parts: the Jupyter Notebook and the command-line application.

<h1>Jupyter Notebook - Image Classifier Project</h1>
This file is a notebook that may be downloaded by the user to train, test and run predictions on images. The training and testing images are already supplied in the flowers directory. If the user would like to add their own images for training, testing or validation, they must ensure that their supplied images are placed in the correct folder. Folders are marked by index (i.e. 1, 2, 3 ,etc. Not my decision, Udacity did it that way :(   ). The user must utilize cat_to_name.json to place their files in the correct folders. The application is not dependant upon how many images there are, so feel free to add as many as you want. 

<h2>How to use:</h2>
<p>1. The notebook is quite straight-forward in its approach. Simply run all of the cells in order, supply a new directory if necessary within the second cell. The notebook expects the directory to be named flowers, within the same directory as the notebook but this may be changed according to needs. It expects the test, validation, and train sets to be named accordingly to the notebook, however these may also be modified.</p>
<p>2. Continue to run cells in order. If the user has a new category json, it may be supplied within the label mapping cell.
<p>3. When building the network, the program will automatically detect a GPU and use it if possible to train. After training, the model will automatically be moved back to the cpu.</p>
<p>4. The notebook will save the checkpoint into the same directory as the notebook. It will also load as well. This will allow the user to train a model and save it to continue training later if necessary.</p>
<p>5. At the end for running predictions, the program will automatically choose random images from the test set supplied in flowers directory, output the prediction label, an image of the file, and a barplot of the top 5 predictions. The notebook may then be converted to an html file.</p> 


<h1> Command-line application - Image Classifier Project </h1>

<h2>Training the model</h2>
This is a command-line application version of the Jupyter Notebook that is a bit more user friendly with several options for the user. The user may provide multiple hyperparameters such as hidden units, epochs, and learning rate. They also have the option to run it on their gpu. To run the training, use the following command example:

```
$python train.py flowers
```

The default architecture is resnet18, learning rate of 0.001, hidden units of 250, and epochs of 1. The program is capable of building architectures of AlexNet, DenseNet, Inception_V3, Resnet and VGG. For those familiar with PyTorch, this is every model from their models library except for SqueezeNet. The following command will create a architecture of DenseNet121, with learning_rate of 0.005, hidden units of 400, epochs of 5 and utilize gpu:

```
$python train.py flowers --arch densenet121 --learning_rate 0.005 --hidden_units 400 --epochs 5 --gpu
```

This program will then output validation loss and test loss every 5 cycles through the epochs as well as output #/5 epochs complete where # is the epoch the model is currently on. Once complete, the program will immediately save a checkpoint within the directory.

<h2>Running Predictions</h2>
To run a prediction on a users image, the user must supply an image directory and a checkopint. The checkpoint is automatically saved within the directory itself, however this option is put in place to allow a user to supply a different checkpoint if desired. Other options include top_k, category_names and gpu. Top_k allows user to choose how many results to print once the model prediction is run. Category names defaults to cat_to_name.json but may be supplied based on different user preferences. Supplying --gpu will allow the user to utilize the gpu on their computer. Example standard input is as follows:

```
$python predict.py flowers/test/1/image_file.jpg
```

Input with options is as follows:

```
$python predict.py flowers/test/1/image_file.jpg --checkpoint checkpoint_file.pth --top_k 10 --category_names cat_to_name_file.json --gpu
```
<h2>Important Notes</h2>
With the flowers directory, if a user supplies their own, it MUST be in the same directory format as supplied or it will not work. Also it must have numbers as file names and a cat_to_name.json must be supplied. Cat_to_name.json must be a dictionary file with the number as the key and label as the value. The way this program is written, you can actually ID whatever you want to as it is not dependant upon the category names being only flowers. It can be used to ID animals, objects or whatever you want as long as the correct directories are supplied.

To run the command-line application, you will need everything in the file except for the jupyter notebook.


<h1>License and Contributions</h1>
This is a project completed by Muhammad Usman as a student for the Udacity NanoDegree- Machine Learning. It is meant to show potential employers Machine Learning skills and will not be further maintained. 

<h2>Flowers File</h2>
Due to file size restrictions on github, the file may not be posted here. I have created a dropbox link to download the flowers dataset <a href='https://www.dropbox.com/sh/j44m1lu8nfb7hf2/AABLzHhZS34DrPVyZgjlgYUKa?dl=0'>here</a>. This same dataset may also be installed from the CNTK Image Datasets through python command by running:

```
$python install_flowers.py
```
