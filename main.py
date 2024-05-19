import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet
from tqdm import tqdm
import time
import os
from sklearn.metrics import precision_recall_fscore_support
from models import CustomModel
from utils import train_one_epoch, validate, train_and_validate





def main():

	data_dir = '/m/triton/scratch/elec/t405-puhe/p/bijoym1/classification/dataset/ISIC_Labelled-20240516T184316Z-001/ISIC_Labelled'

	print("Training conv layers of the base model too")

	input_shape = (300, 300)
	num_classes = 8
	epochs = 30
	batch_size = 64
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_transforms = {
	    'train': transforms.Compose([
		transforms.RandomResizedCrop(input_shape, scale=(0.08, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ]),
	    'val': transforms.Compose([
		transforms.Resize(input_shape),
		transforms.CenterCrop(input_shape),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ]),
	    'test': transforms.Compose([
		transforms.Resize(input_shape),
		transforms.CenterCrop(input_shape),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ])
	}

	base_model = efficientnet.efficientnet_b0(pretrained=True)
	model = CustomModel(base_model, num_classes)
	model.to(device)

	print(f"model architecture\n{model}\n{'*'*20}\n\n")

	# Define optimizer, learning rate scheduler, and early stopping
	optimizer = optim.Adam(model.parameters(), lr=0.0001)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
	criterion = nn.CrossEntropyLoss()

	file_path = '/m/triton/scratch/elec/t405-puhe/p/bijoym1/classification/model_v3.pth'
	if os.path.exists(file_path):
	    checkpoint = torch.load(file_path)
	    model.load_state_dict(checkpoint['model_state_dict'])
	    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	    print("Model checkpoint has been loaded")

	# Splitting the dataset
	full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
	train_size = int(0.8 * len(full_dataset))
	val_size = len(full_dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

	# Creating data loaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs)
	
	
	
	
	
if __name__ == "__main__":
	main()


















