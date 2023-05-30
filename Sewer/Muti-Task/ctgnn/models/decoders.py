import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_layers import GCNLayer, GATLayer
from .graph_normalization import symmetric_matrix_normalization, in_degree_matrix_normalization



class SimpleHead(nn.Module):

    def __init__(self, n_tasks, num_classes, input_channels, decoder_channels, pool = "Avg", list_input=False, **kwargs):
        
        super(SimpleHead, self).__init__()

        assert(isinstance(n_tasks, (float, int)))

        self.n_tasks = n_tasks
        self.num_classes = num_classes
        self.pool = pool
        self.list_input = list_input

        if self.pool == "Avg":
            self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool == "Max":
            self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        else: 
            self.global_pooling = nn.Identity()

        if self.list_input:
            self.forward_func = self.forward_list
        else:
            self.forward_func = self.forward_tensor

        # Construct the independent decoder heads
        self.construct_decoder_heads(input_channels, decoder_channels)


    def construct_decoder_heads(self, input_channels, decoder_channels):
        # Initial MLP layers
        self.mlp = nn.ModuleList()
        self.heads = nn.ModuleList()

        if decoder_channels is None or len(decoder_channels) == 0:
            # In case the extracted features connect directly to the classification layers
            self.penultimate_channels = input_channels

            for _ in range(self.n_tasks):
                self.mlp.append(nn.Sequential())
        else:
            # In case there are intermediate layers between the extracted feature and classification layers
            self.penultimate_channels = decoder_channels[-1]

            for _ in range(self.n_tasks):
                task_head = []
                for idx, _ in enumerate(decoder_channels):
                    if idx == 0:
                        task_head.append(nn.Linear(input_channels, decoder_channels[0]))
                        task_head.append(nn.ReLU(inplace=True))
                    else:
                        task_head.append(nn.Linear(decoder_channels[idx-1], decoder_channels[idx]))
                        task_head.append(nn.ReLU(inplace=True))
                
                self.mlp.append(nn.Sequential(*task_head))

        for task in range(self.n_tasks):
            task_classifier = [nn.Linear(self.penultimate_channels, self.num_classes[task])]
            self.heads.append(nn.Sequential(*task_classifier))


    def forward(self, x):
        feats = self.forward_func(x)
        return feats, feats
        
    def forward_list(self, x):
        out = []
        for idx in range(self.n_tasks):
            xtmp = self.global_pooling(x[idx])
            xtmp = torch.flatten(xtmp, 1)
            xtmp = self.mlp[idx](xtmp) 
            out.append(self.heads[idx](xtmp))

        return out
        
    def forward_tensor(self, x):
        out = []
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        for idx in range(self.n_tasks):
            xtmp = self.mlp[idx](x) 
            out.append(self.heads[idx](xtmp))

        return out



class CTGNN(nn.Module):
    def __init__(self, n_tasks, num_classes, input_channels, decoder_channels, adj_mat_path, gnn_head, gnn_layers = 1, gnn_channels = 128, gnn_dropout=0.6, gnn_residual = True, attention_heads=None, bottleneck_channels=32, adj_normalization="", pool = "Avg", list_input=False, shared_bottleneck = False, shared_linear = True, gnn_residual_act = "Pre", **kwargs):
        
        super(CTGNN, self).__init__()

        assert(isinstance(n_tasks, (float, int)))

        self.n_tasks = n_tasks
        self.num_classes = num_classes
        self.pool = pool
        self.list_input = list_input
        
        if isinstance(gnn_channels, (list, tuple)):
            print("Using just the first index of {} i.e. {}".format(gnn_channels, gnn_channels[0]))
            self.gnn_channels = gnn_channels[0]
        else:
            self.gnn_channels = gnn_channels

        if self.pool == "Avg":
            self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool == "Max":
            self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        else: 
            self.global_pooling = nn.Identity()

        if self.list_input:
            self.forward_func = self.forward_list
        else:
            self.forward_func = self.forward_tensor        

        # Load and normalize the adjacency matrix
        self.construct_adjacency_matrix(adj_mat_path, adj_normalization)

        # Construct the independent decoder heads
        self.construct_decoder_heads(input_channels, decoder_channels)

        # Construct the bottleneck projection layer
        self.shared_bottleneck = shared_bottleneck
        self.bottleneck_channels = bottleneck_channels

        self.construct_bottlenecks()

        # Construct the per-class up-projection layer - creates the node embeddings
        self.construct_feature_projections()

        # Construct the Cross-Task Graph Neural Networks (CT-GNN)
        self.gnn_layers = gnn_layers
        self.attention_heads = attention_heads
        self.gnn_head = gnn_head
        self.gnn_residual = gnn_residual
        self.gnn_residual_act = gnn_residual_act
        self.gnn_dropout = gnn_dropout
        self.alpha = 0.2

        self.construct_ct_gnn()
        
        # Construct the linear prediction layer, with either shared or per-class layer
        self.shared_linear = shared_linear

        self.construct_linear_projection()



    def construct_decoder_heads(self, input_channels, decoder_channels):
        # Initial MLP layers
        self.mlp = nn.ModuleList()
        self.heads = nn.ModuleList()

        if decoder_channels is None or len(decoder_channels) == 0:
            # In case the extracted features connect directly to the classification layers
            self.penultimate_channels = input_channels

            for _ in range(self.n_tasks):
                self.mlp.append(nn.Sequential())
        else:
            # In case there are intermediate layers between the extracted feature and classification layers
            self.penultimate_channels = decoder_channels[-1]

            for _ in range(self.n_tasks):
                task_head = []
                for idx, _ in enumerate(decoder_channels):
                    if idx == 0:
                        task_head.append(nn.Linear(input_channels, decoder_channels[0]))
                        task_head.append(nn.ReLU(inplace=True))
                    else:
                        task_head.append(nn.Linear(decoder_channels[idx-1], decoder_channels[idx]))
                        task_head.append(nn.ReLU(inplace=True))
                
                self.mlp.append(nn.Sequential(*task_head))

        for task in range(self.n_tasks):
            task_classifier = [nn.Linear(self.penultimate_channels, self.num_classes[task])]
            self.heads.append(nn.Sequential(*task_classifier))


    def construct_bottlenecks(self):
        if self.bottleneck_channels is None:
            self.bottleneck_channels = self.gnn_channels//4

        if self.shared_bottleneck:
            num_bottlenecks = 1
        else:
            num_bottlenecks = self.n_tasks

        self.bottlenecks = nn.ModuleList()
        for _ in range(num_bottlenecks):
            btl_neck = []
            btl_neck.append(nn.Linear(self.penultimate_channels, self.bottleneck_channels, bias = True))
            btl_neck.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            self.bottlenecks.append(nn.Sequential(*btl_neck))


    def construct_adjacency_matrix(self, adj_mat_path, adj_normalization):
        
        ## GNN Adj Mat
        if isinstance(adj_mat_path, torch.Tensor):
            adj_mat = adj_mat_path.numpy()
        elif isinstance(adj_mat_path, np.ndarray):
            adj_mat = adj_mat_path
        else:
            if os.path.isfile(adj_mat_path):
                adj_mat = np.load(adj_mat_path)
            else:
                raise ValueError("The provided adjacency matrix file does not exists: {}".format(adj_mat_path))
        
        if adj_normalization == "In":
            adj_mat = in_degree_matrix_normalization(adj_mat)
        elif adj_normalization == "Sym":
            adj_mat = symmetric_matrix_normalization(adj_mat)
        elif adj_normalization == "":
            adj_mat = adj_mat
        else:
            raise ValueError("The provided adjacency normalization method is not available: {}".format(adj_normalization))
        adj_mat = torch.from_numpy(adj_mat).float()

        self.register_buffer("adj", adj_mat)


    def construct_feature_projections(self): 
        self.feat_projs = nn.ModuleList()
        for idx in range(len(self.num_classes)):
            self.feat_projs.append(nn.Sequential(*[nn.Linear(self.bottleneck_channels, self.gnn_channels*self.num_classes[idx], bias = True), nn.LeakyReLU(inplace=True, negative_slope=0.2)]))


    def construct_ct_gnn(self):

        if self.gnn_head == "GCN":
            self.ct_gnn = nn.ModuleList([GCNLayer(in_features=self.gnn_channels, dropout = self.gnn_dropout, alpha = self.alpha, residual = self.gnn_residual, residual_act = self.gnn_residual_act) for _ in range(self.gnn_layers)])
        elif self.gnn_head == "GAT":
            self.ct_gnn = nn.ModuleList([GATLayer(in_features=self.gnn_channels, attention_heads=self.attention_heads, dropout = self.gnn_dropout, alpha = self.alpha, residual = self.gnn_residual, residual_act = self.gnn_residual_act, concat=True) for _ in range(self.gnn_layers)])
        else:
            raise ValueError("The provided GNN is not a valid: {}".format(self.gnn_head))


    def construct_linear_projection(self):
        if self.shared_linear:
            self.linear_class = nn.Sequential(nn.Linear(self.gnn_channels, 1, bias=True))
        else:
            self.linear_class = nn.ModuleList([nn.Linear(self.gnn_channels, 1, bias=True) for _ in range(sum(self.num_classes))])



    def forward(self, x):
        _adj = self.adj.detach()

        node_feats, feats = self.forward_func(x)
        node_feats = torch.stack(node_feats, dim=1)
        
        for layer in range(self.gnn_layers):
            node_feats = self.ct_gnn[layer](node_feats, adj=_adj)

        if self.shared_linear:
            node_feats = self.linear_class(node_feats)
        else:
            out = [self.linear_class[idx](node_feats[:,idx]) for idx in range(sum(self.num_classes))]
            node_feats = torch.stack(out, dim=1)
        node_feats = node_feats.squeeze(-1)

        return torch.split(node_feats, self.num_classes, dim=1), feats

        
    def forward_list(self, x):
        out = []
        feats = []

        for idx in range(self.n_tasks):
            if self.shared_bottleneck:
                btl_idx = 0
            else:
                btl_idx = idx

            xtmp = self.global_pooling(x[idx])
            xtmp = torch.flatten(xtmp, 1)

            xtmp = self.mlp[idx](xtmp) 
            feats.append(self.heads[idx](xtmp))

            # per-task down projection and per node up projection
            xtmp = self.bottlenecks[btl_idx](xtmp)
            xtmp = self.feat_projs[idx](xtmp)
            out.extend(torch.split(xtmp, self.gnn_channels, dim=1))
        return out, feats
        
    def forward_tensor(self, x):
        out = []
        feats = []
        x = self.global_pooling(x)
        x = torch.flatten(x, 1)
        for idx in range(self.n_tasks):
            if self.shared_bottleneck:
                btl_idx = 0
            else:
                btl_idx = idx

            xtmp = self.mlp[idx](x) 
            feats.append(self.heads[idx](xtmp))
            
            # per-task down projection and per node up projection
            xtmp = self.bottlenecks[btl_idx](xtmp)
            xtmp = self.feat_projs[idx](xtmp)
            out.extend(torch.split(xtmp, self.gnn_channels, dim=1))
        return out, feats