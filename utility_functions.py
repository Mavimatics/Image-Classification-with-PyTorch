import torch
from torchvision import transforms
from torch import nn
from PIL import Image

def load_data(data_directory):
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(root=data_directory + '/' + x, transform=data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                   for x in ['train', 'valid', 'test']}

    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], class_to_idx

def build_model(arch, hidden_units, class_to_idx):
    # Build the model
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Invalid architecture. Supported architectures: resnet50, vgg16")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, len(class_to_idx)),
        nn.LogSoftmax(dim=1)
    )

    model.fc = classifier

    return model

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, use_gpu):
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    model.to(device)
    
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 5


    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        
            # Move input and label tensors to GPU
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
            
                # turn off gradients to speedup the code
                #with torch.no_grad():
                
                # validation pass here
                
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    
                    valid_loss += loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    

def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier
    }

    # Save the checkpoint
    checkpoint_filepath = f"{save_dir}/checkpoint.pth"
    torch.save(checkpoint, checkpoint_filepath)

    print(f"Checkpoint saved at {checkpoint_filepath}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Load the pre-trained model
    if checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Invalid architecture in the checkpoint. Supported architectures: resnet50, vgg16")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Modify the classifier based on the checkpoint
    model.fc = checkpoint['classifier']
    
    # Load the model state dictionary
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load class_to_idx mapping
    model.class_to_idx = checkpoint.get('class_to_idx', None)
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
  
    # Open the image using PIL
    img = Image.open(image_path)

    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply the transformations
    img_tensor = preprocess(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Preprocess the image
    img_tensor = process_image(image_path)
    
    # Move the input tensor to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        
    # Calculate probabilities and classes
    probs, indices = torch.topk(torch.softmax(output, dim=1), topk)
    
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    return probs[0].tolist(), classes

