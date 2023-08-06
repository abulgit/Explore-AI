import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, cnn_pretrained=True):
        super(CNNLSTMModel, self).__init__()
        # Initialize the CNN model for image feature extraction
        self.cnn = models.resnet50(pretrained=cnn_pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Initialize the LSTM model for caption generation
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        # Extract image features using the CNN model
        with torch.no_grad():
            features = self.cnn(images)
        features = features.view(features.size(0), 1, -1)
        
        # Generate word embeddings for captions
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features, embeddings), dim=1)
        
        # Pass the embeddings through the LSTM model
        lstm_out, _ = self.lstm(embeddings)
        
        # Predict the next word using the linear layer
        outputs = self.linear(lstm_out)
        return outputs

def create_cnn_lstm_model(vocab_size, embed_size, hidden_size, num_layers, cnn_pretrained=True):
    return CNNLSTMModel(vocab_size, embed_size, hidden_size, num_layers, cnn_pretrained=cnn_pretrained)
