import torch
import torch.nn as nn
import torch.nn.functional as F



class GCNLayer(nn.Module):
    """
    Simple GCN layer.
    Implementation based on: https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, dropout, alpha, residual = False, residual_act = "Pre", bias=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))

        self.dropout = dropout
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=True)

        self.residual = residual
        self.res_pre_act = False
        self.res_post_act = False
        
        if residual_act == "Pre" or residual_act == "Both":
            self.res_pre_act = True
        if residual_act == "Post" or residual_act == "Both":
            self.res_post_act = True

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        
        # Skip connection useed by Kipf et al.
        if self.res_pre_act:
            output = self.leakyrelu(output)
            output = F.dropout(output, self.dropout, training=self.training)

        if self.residual:
            output = output + input
        
        ## Original ResNet Structure
        if self.res_post_act:
            output = self.leakyrelu(output)
            output = F.dropout(output, self.dropout, training=self.training)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GATLayer(nn.Module):
    """
    Simple GAT layer.
    Implementation based on https://github.com/Diego999/pyGAT/issues/36
    """

    def __init__(self, in_features, attention_heads, dropout, alpha, residual = False, residual_act = "Pre", concat=True):
        super(GATLayer, self).__init__()

        self.attention_heads = attention_heads
        self.in_features = in_features
        self.out_features = in_features // self.attention_heads

        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=True)

        self.residual = residual
        self.res_pre_act = False
        self.res_post_act = False
        if residual_act == "Pre" or residual_act == "Both":
            self.res_pre_act = True
        if residual_act == "Post" or residual_act == "Both":
            self.res_post_act = True
            

        self.weight = nn.Parameter(torch.Tensor(size=(self.attention_heads, self.in_features, self.out_features)))
        self.att = nn.Parameter(torch.Tensor(size=(self.attention_heads, 2 * self.out_features, 1)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def forward(self, input, adj):
        adj = adj.detach()

        att_out = [self.attention_forward(input, adj, att_head) for att_head in range(self.attention_heads)]
        
        if self.concat:
            out = torch.cat(att_out, dim=-1)

            
            if self.res_pre_act:              
                out = F.elu(out)
                out = F.dropout(out, self.dropout, training=self.training)

            if self.residual:
                out = out + input

            if self.res_post_act:
                out = F.elu(out)
                out = F.dropout(out, self.dropout, training=self.training)

            return out
        else:
            return torch.mean(torch.stack(att_out), dim=0)

    def attention_forward(self, input, adj, head_idx):
        Wh = torch.matmul(input, self.weight[head_idx])

        a_input = self.batch_prepare_attentional_mechanism_input(Wh)

        e = torch.matmul(a_input, self.att[head_idx]).squeeze(-1)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features)+" x "+str(self.attention_heads) + ' -> ' +  str(self.in_features)  + ')'