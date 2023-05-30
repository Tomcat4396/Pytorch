import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_layers import GraphConvolutionLayer, GraphAttentionLayer


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, adj_mat, in_features, hidden_dims, out_features, num_layers, dropout, lrelu_slope = 0.2, gnn_residual = False):
        super(GraphConvolutionNetwork, self).__init__()
        assert adj_mat.shape[0] == adj_mat.shape[1], "The adjacency matrix is not square: Adj Mat {}".format( adj_mat.shape)

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.lrelu_slope = lrelu_slope

        self.gnn_residual = gnn_residual

        self.register_buffer("adj", adj_mat)

        if isinstance(self.hidden_dims, (float, int)):
            self.hidden_dims = [int(self.hidden_dims)]

        if self.num_layers > len(self.hidden_dims):
            assert len(self.hidden_dims) == 1, "More hidden dimensions defined than number of layers: {} layers - {} hidden dimensions".format(self.num_layers, self.hidden_dims)
            self.hidden_dims = self.hidden_dims * self.num_layers
        else:
            assert len(self.hidden_dims) == self.num_layers, "More hidden dimensions defined than number of layers: {} layers - {} hidden dimensions".format(self.num_layers, self.hidden_dims)

        self.gcn = nn.ModuleList()

        self.gcn.append(GraphConvolutionLayer(self.in_features, self.hidden_dims[0]))
        for layer in range(1, self.num_layers):
            self.gcn.append(GraphConvolutionLayer(self.hidden_dims[layer-1], self.hidden_dims[layer], residual = self.gnn_residual))
        self.gcn.append(GraphConvolutionLayer(self.hidden_dims[-1], self.out_features))

        self.leaky_relu = nn.LeakyReLU(self.lrelu_slope, inplace=True)
        
        assert len(self.gcn) == (self.num_layers+1)
    
    def forward(self, x):
        _adj = self.adj.detach()
        for layer in range(self.num_layers):
            x = self.gcn[layer](x, _adj)
            x = self.leaky_relu(x)
            F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, _adj)
        
        return x


class GraphAttentionNetwork(nn.Module):
    def __init__(self, adj_mat, in_features, hidden_dims, out_features, num_layers, num_heads, dropout, lrelu_slope = 0.2):
        super(GraphAttentionNetwork, self).__init__()
        assert adj_mat.shape[0] == adj_mat.shape[1], "The adjacency matrix is not square: Adj Mat {}".format( adj_mat.shape)

        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.num_layers = num_layers
        self.lrelu_slope = lrelu_slope
        self.num_heads = num_heads
        self.dropout = dropout

        self.register_buffer("adj", adj_mat)

        if isinstance(self.hidden_dims, (float, int)):
            self.hidden_dims = [int(self.hidden_dims)]
        if isinstance(self.num_heads, (float, int)):
            self.num_heads = [int(self.num_heads)]

        if self.num_layers > len(self.hidden_dims):
            assert len(self.hidden_dims) == 1, "More hidden dimensions defined than number of layers: {} layers - {} hidden dimensions".format(self.num_layers, self.hidden_dims)
            self.hidden_dims = self.hidden_dims * self.num_layers
        else:
            assert len(self.hidden_dims) == self.num_layers, "More hidden dimensions defined than number of layers: {} layers - {} hidden dimensions".format(self.num_layers, self.hidden_dims)

    
        if self.num_layers > len(self.num_heads):
            assert len(self.num_heads) == 1, "More attention heads defined than number of layers: {} layers - {} Attention head".format(self.num_layers, self.num_heads)
            self.num_heads = self.num_heads * self.num_layers
        else:
            assert len(self.num_heads) == self.num_layers, "More attention heads defined than number of layers: {} layers - {} Attention heads".format(self.num_layers, self.num_heads)


        self.gat = nn.ModuleList()
        self.gat.append(nn.ModuleList([GraphAttentionLayer(self.in_features, self.hidden_dims[0]//self.num_heads[0], dropout=dropout, alpha=self.lrelu_slope, concat=True) for _ in range(self.num_heads[0])]))
        for layer in range(1, self.num_layers):
            self.gat.append(nn.ModuleList([GraphAttentionLayer(self.hidden_dims[layer-1], self.hidden_dims[layer]//self.num_heads[layer], dropout=dropout, alpha=self.lrelu_slope, concat=True) for _ in range((self.num_heads[layer]))]))
        self.gat.append(nn.ModuleList([GraphAttentionLayer(self.hidden_dims[-1], self.out_features, dropout=dropout, alpha=self.lrelu_slope, concat=False) for _ in range((self.num_heads[-1]))]))
    
        assert len(self.gat) == (self.num_layers+1)
    
    def forward(self, x):
        _adj = self.adj.detach()

        for layer in range(self.num_layers):
            x = torch.cat([att(x, _adj) for att in self.gat[layer]], dim=-1)
            F.dropout(x, self.dropout, training=self.training)

        x = [att(x, _adj) for att in self.gat[-1]]
        x = torch.mean(torch.stack(x), dim=0)
        return x