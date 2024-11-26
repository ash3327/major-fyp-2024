import numpy as np
from sklearn.decomposition import PCA
import json

class PCAVisualizer:
    def __init__(self, class_means_path):
        """Initialize PCA visualizer with class means"""
        with open(class_means_path, "r") as f:
            self.class_means = json.load(f)
        
        # Convert class means to numpy array for PCA
        self.mean_embeddings = np.array([mean for mean in self.class_means.values()])
        self.labels = list(self.class_means.keys())
        
        # Fit PCA on mean embeddings
        self.pca = PCA(n_components=2)
        self.pca.fit(self.mean_embeddings)
        
        # Transform mean embeddings
        self.transformed_means = self.pca.transform(self.mean_embeddings)
        
    def transform_embedding(self, embedding):
        """Transform a new embedding using the fitted PCA"""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return self.pca.transform(embedding)
    
    def get_visualization_data(self, current_embedding=None):
        """Get data for visualization including means and current point if provided"""
        data = {
            'means': {
                'x': self.transformed_means[:, 0].tolist(),
                'y': self.transformed_means[:, 1].tolist(),
                'labels': self.labels
            }
        }
        
        if current_embedding is not None:
            transformed_current = self.transform_embedding(current_embedding)
            data['current'] = {
                'x': transformed_current[0, 0],
                'y': transformed_current[0, 1]
            }
            
        return data
