import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from pydantic.dataclasses import dataclass

from torch_geometric.nn import RGCNConv, TransformerConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.aggr import Set2Set
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class RGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, graph_pooling, loss_function):
        super(RGCN, self).__init__() 
        torch.manual_seed(10)
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(input_channels,100,num_relations=3))
        self.convs.append(RGCNConv(100,hidden_channels,num_relations=3))
        for _ in range(2): self.convs.append(RGCNConv(hidden_channels,hidden_channels,num_relations=3))
        self.norms  = torch.nn.ModuleList()
        self.norms.append(BatchNorm(100))
        self.norms.append(BatchNorm(hidden_channels))
        for _ in range(2): self.norms.append(BatchNorm(hidden_channels))

        for i in range(4): init.xavier_uniform_(self.convs[i].weight)

        if graph_pooling == "sum": self.pool = global_add_pool
        elif graph_pooling == "mean": self.pool = global_mean_pool
        elif graph_pooling == "max": self.pool = global_max_pool
        elif graph_pooling == "set2set": self.pool = Set2Set(hidden_channels, processing_steps=2)
        else: raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set": self.lin1 = nn.Linear(2*hidden_channels + 38, 100)
        else: self.lin1 = nn.Linear(hidden_channels + 38, 100)
        self.lin2 = nn.Linear(100, 1)

    def forward(self, x, edge_attr, edge_index, edge_type, MetaData, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
        x = self.pool(x, batch)
        x = torch.cat([x, MetaData], dim=1)
        x = self.lin1(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.lin2(x)
        return x


class RGCN_without_metadata(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, graph_pooling, loss_function):
        super(RGCN_without_metadata, self).__init__() 
        torch.manual_seed(10)
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(input_channels,100,num_relations=3))
        self.convs.append(RGCNConv(100,hidden_channels,num_relations=3))
        for _ in range(2): self.convs.append(RGCNConv(hidden_channels,hidden_channels,num_relations=3))
        self.norms  = torch.nn.ModuleList()
        self.norms.append(BatchNorm(100))
        self.norms.append(BatchNorm(hidden_channels))
        for _ in range(2): self.norms.append(BatchNorm(hidden_channels))

        for i in range(4): init.xavier_uniform_(self.convs[i].weight)

        if graph_pooling == "sum": self.pool = global_add_pool
        elif graph_pooling == "mean": self.pool = global_mean_pool
        elif graph_pooling == "max": self.pool = global_max_pool
        elif graph_pooling == "set2set": self.pool = Set2Set(hidden_channels, processing_steps=2)
        else: raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set": self.lin1 = nn.Linear(2*hidden_channels, 100)
        else: self.lin1 = nn.Linear(hidden_channels, 100)
        self.lin2 = nn.Linear(100, 1)

    def forward(self, x, edge_attr, edge_index, edge_type, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, edge_type))
        x = self.pool(x, batch)
        x = self.lin1(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.lin2(x)
        return x