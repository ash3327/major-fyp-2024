import os
import torch

# Function to save the model
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

# Function to load the model
def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set the model to evaluation mode
    print(f'Model loaded from {file_path}')
    return model
