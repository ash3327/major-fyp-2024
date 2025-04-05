"""
Source: 
1. https://github.com/pyg-team/pytorch_geometric/discussions/9531
2. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
3. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('.')

from torch_geometric.nn import GATConv, BatchNorm, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch

from training.contrastive.preprocess import get_6dof
from training.contrastive import topology

def graph_transform(batch_list):
    return Batch.from_data_list([Data(x=batch, edge_index=topology.edge_indices) for batch in batch_list])

class HandEncoderGCN3dof(nn.Module):
    def __init__(
            self, 
            node_in_channels=3, 
            hidden_channels=64, 
            num_layers=3,
            num_heads=4,
            embedding_size=128, 
            dropout_rate=0.3, 
            leaky_slope=0.01,
            pooling_method: str = 'mean',
            edge_index=None,
            fn=graph_transform
        ):
        """
        edge_index: [2, num edges] storing the graph connectivity
        fn: function that should be applied before each pass
        """
        super().__init__()

        # --- check ---
        if pooling_method not in ['mean', 'max', 'add']:
            raise ValueError("pooling_method must be 'mean', 'max', or 'add'")
        
        # --- params ---
        self.output_graph_features = False
        self.fn = fn

        self.node_in_channels = node_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.leaky_slope = leaky_slope
        self.embedding_size = embedding_size

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # --- Edge indices ---
        if edge_index is None:
            edge_index = topology.edge_indices
        self.edge_index = edge_index

        # --- Layers ---
        current_dim = node_in_channels

        # GCN layers
        for i in range(num_layers):
            in_dim = current_dim
            concat = True # Keep concat=True for multi-head stability
            out_dim = hidden_channels

            conv = GCNConv(in_channels=in_dim, out_channels=out_dim)
            self.gcn_layers.append(conv)

            # bn_dim =  * num_heads if concat else out_dim
            self.batch_norms.append(BatchNorm(out_dim))
            current_dim = out_dim

        # Pooling layer
        if pooling_method == 'mean': self.pool = global_mean_pool
        elif pooling_method == 'max': self.pool = global_max_pool
        else: self.pool = global_add_pool

        # MLP head
        mlp_in_dim = hidden_channels
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(negative_slope=leaky_slope),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_channels, embedding_size)
        )

    def set_output_graph_features(self, value:bool=True):
        """Force the model to output also the graph features after the last GAT layer."""
        self.output_graph_features = value

    def forward(self, data):
        """Forward pass: apply 6DOF preprocessing, flatten input, compute embeddings, and normalize."""
        # x = x.view(x.size(0), 21, 3)  # Reshape to [B*Grid_size, 21, 3]
        # x = get_6dof(x).to(device=x.device)  # Compute 6DOF embeddings
        # print(f"Type {type(data)}, {data}")
        # print(data.shape)
        if self.fn:
            data = data.view(-1, 21, 3)
            data = self.fn(data)
        x = data.x.view(-1,self.node_in_channels) # Reshape to [B*G,N,3] -> [B*N, 3]
        edge_index = torch.tensor(data.edge_index).to(x.device) 
        edge_index = edge_index.view(-1,2).T # [B*G,E,2] -> [B*G*E,2] -> [2, B*G*E]
        # print(data.x.shape,x.shape,len(edge_index),edge_index.shape)
        # Pass through GAT layers
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index) # No edge_attr needed
            x = self.batch_norms[i](x)
            x = F.leaky_relu(x, negative_slope=self.leaky_slope)

        # Pooling
        features = x
        pooled_x = self.pool(x, batch=data.batch) # this would make it back

        # MLP head
        embedding = self.mlp_head(pooled_x) # [B*G, D]

        if self.output_graph_features:
            return embedding, features
        return embedding
    
class HandEncoderGCN6dof(HandEncoderGCN3dof):
    def __init__(self, *args, fn=graph_transform, **kwargs):
        kwargs['node_in_channels'] = 6
        kwargs['fn'] = None
        self.fn2 = fn
        super().__init__(*args, **kwargs)

    def forward(self, data):
        if self.fn2:
            data = data.view(-1, 21, 3)
            data = self.fn2(data)
        x = data.x.view(-1,21,3)
        x = get_6dof(x).to(device=x.device)
        data.x = x.view(-1,6)
        return super().forward(data)

    