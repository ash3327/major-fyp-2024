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
        
        # Initialize PCA with more components than needed to handle new classes
        self.pca = PCA(n_components=2)
        self.pca.fit(self.mean_embeddings)
        
        # Transform mean embeddings
        self.transformed_means = self.pca.transform(self.mean_embeddings)
        
        # Store custom classes
        self.custom_classes = {}
        self.custom_transformed = None
    
    def add_class(self, class_name, embedding):
        """Add a new class with its embedding"""
        self.custom_classes[class_name] = embedding
        
        # Update PCA with all embeddings (original + custom)
        all_embeddings = np.vstack([
            self.mean_embeddings,
            np.array([emb for emb in self.custom_classes.values()])
        ])
        
        # Refit PCA and transform all points
        self.pca.fit(all_embeddings)
        self.transformed_means = self.pca.transform(self.mean_embeddings)
        self.custom_transformed = self.pca.transform(
            np.array([emb for emb in self.custom_classes.values()])
        )
    
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
        
        # Add custom classes if they exist
        if self.custom_transformed is not None:
            data['custom_means'] = {
                'x': self.custom_transformed[:, 0].tolist(),
                'y': self.custom_transformed[:, 1].tolist(),
                'labels': list(self.custom_classes.keys())
            }
        
        if current_embedding is not None:
            transformed_current = self.transform_embedding(current_embedding)
            data['current'] = {
                'x': transformed_current[0, 0],
                'y': transformed_current[0, 1]
            }
            
        return data
