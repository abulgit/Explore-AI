import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from models.image_captioning import create_cnn_lstm_model
from utils.evaluation import evaluate_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
vocab_size = 10000  # Adjust based on your dataset vocabulary size
embed_size = 256
hidden_size = 512
num_layers = 1
cnn_pretrained = True
batch_size = 32

# Load the test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

test_dataset = CaptionDataset('data/images/', 'data/captions.txt', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create the model
model = create_cnn_lstm_model(vocab_size, embed_size, hidden_size, num_layers, cnn_pretrained)
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Evaluate the model
bleu_score, meteor_score, cider_score = evaluate_model(model, test_loader, vocab_size)
print(f"BLEU Score: {bleu_score:.4f}")
print(f"METEOR Score: {meteor_score:.4f}")
print(f"CIDEr Score: {cider_score:.4f}")
