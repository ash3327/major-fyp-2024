from ..model_loader import load_model as _load_model
from ..verifiers.verifier import extract_features as _extract_features

class ModelLib:
    def __init__(self, configs):
        print(configs)
        
    def load(self, device='cuda'):
        self.model = SimpleClassifier(num_classes=27)
        self.model = _load_model(self.model, f'{ROOT}/saved_models/model-2024-10-28-1.pth')
        self.model.to(device)

    def extract_features(self, dataloader, device='cuda'):
        """
        :param iterator dataloader: Dataloader or iterator that returns 
        """
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        features_list = []
        labels_list = []

        with torch.no_grad(): 
            for feats in tqdm(dataloader):
                inputs = feats[0]
                labels = feats[1] if len(feats) > 1 else None
                inputs = inputs.to(device)
                _, last_layer_input = self.model(inputs)  # Get last layer input
                features_list.append(last_layer_input.cpu().numpy())  # Store features
                labels_list.append(labels)  # Store labels

        return np.concatenate(features_list), np.concatenate(labels_list)

    
