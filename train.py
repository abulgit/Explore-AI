import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from models.image_captioning import create_cnn_lstm_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
vocab_size = 10000  # Adjust based on your dataset vocabulary size
embed_size = 256
hidden_size = 512
num_layers = 1
cnn_pretrained = True
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = CaptionDataset('data/images/', 'data/captions.txt', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Create the model
model = create_cnn_lstm_model(vocab_size, embed_size, hidden_size, num_layers, cnn_pretrained)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        
        # Forward pass
        outputs = model(images, captions[:, :-1])
        targets = captions[:, 1:].contiguous().view(-1)
        loss = criterion(outputs.view(-1, vocab_size), targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
print("Training finished. Model saved.")
