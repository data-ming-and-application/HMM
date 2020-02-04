import torch
import torch.nn as nn
import torch.nn.functional as F


class HMM(nn.Module):

    def __init__(self,
                 v1_dim,
                 v2_dim,
                 v1_sample_num,
                 v2_sample_num,
                 hidden_dim,
                 v1_edge_attr_dim,
                 v2_edge_attr_dim,
                 edge_layer_num,
                 edge_head_num,
                 edge_type_num,
                 edge_emb_dim,
                 label_num):
        super(HMM, self).__init__()
        self.v1_dim = v1_dim
        self.v2_dim = v2_dim
        self.label_num = label_num
        self.hidden_dim = hidden_dim
        self.edge_emb = nn.Embedding(edge_type_num, edge_emb_dim)
        self.v1_map_layer = nn.Linear(v1_dim, hidden_dim)
        self.v2_map_layer = nn.Linear(v2_dim, hidden_dim)
        self.edge_layer_num = edge_layer_num
        self.edge_head_num = edge_head_num
        self.v1_edge_layers = nn.ModuleList()
        self.v2_edge_layers = nn.ModuleList()
        self.v1_edge_attr_dim = v1_edge_attr_dim
        self.v2_edge_attr_dim = v2_edge_attr_dim
        self.classify_layer = nn.Linear(hidden_dim*2, label_num)
        for i in range(edge_layer_num):
            self.v1_edge_layers.append(nn.Linear(edge_emb_dim+v1_edge_attr_dim,
                                                 edge_head_num))
            self.v2_edge_layers.append(nn.Linear(edge_emb_dim+v2_edge_attr_dim,
                                                 edge_head_num))
        self.drop = nn.Dropout(p=0.5)
        self.mm_layer = nn.Linear(hidden_dim*2, 1)

    # nodes: [batch_size, v1_dim]
    # v1nodes: [batch_size, v1_sample_num, v1_dim]
    # v2nodes: [batch_size, v2_sample_num, v2_dim]
    # v1adj: [batch_size, v1_sample_num+1, v1_sample_num+1]
    # v2adj: [batch_size, v2_sample_num+1, v2_sample_num+1]
    # v1edge_attr: [batch_size, v1_sample_num+1, v1_sample_num+1]
    # v2edge_attr: [batch_size, v2_sample_num+1, v2_sample_num+1]

    def forward(self,
                nodes,
                v1nodes,
                v2nodes,
                v1adj,
                v2adj,
                v1edge_attr,
                v2edge_attr):
        batch_size = nodes.shape[0]
        v1map = self.v1_map_layer(nodes)
        v1nodes = self.v1_map_layer(v1nodes)
        v2nodes = self.v2_map_layer(v2nodes)
        nodes = v1map.view((batch_size, 1, self.hidden_dim))
        v1nodes = v1nodes.view((batch_size, -1, self.hidden_dim))
        v1nodes = torch.cat([nodes, v1nodes], dim=1)
        v2nodes = torch.cat([nodes, v2nodes], dim=1)
        v1_num = v1nodes.shape[1]
        v2_num = v2nodes.shape[1]
        v1_dim = v1nodes.shape[2]
        v2_dim = v2nodes.shape[2]
        v1_edge_emb = self.edge_emb(v1adj)
        v2_edge_emb = self.edge_emb(v2adj)
        v1edge_attr = v1edge_attr.view(*v1edge_attr.shape, 1)
        v2edge_attr = v2edge_attr.view(*v2edge_attr.shape, 1)

        if self.v1_edge_attr_dim > 0:
            v1_edge_emb = torch.cat([v1_edge_emb, v1edge_attr], dim=3)
        if self.v2_edge_attr_dim > 0:
            v2_edge_emb = torch.cat([v2_edge_emb, v2edge_attr], dim=3)
        # v1nodes: [batch_size, v1_sample_num, emb_dim]
        # v1_edge_emb, v2_edge_emb:
        # [batch_size, v1_sample_num+1, v1_sample_num+1, edge_emb_dim]

        for i in range(self.edge_layer_num):
            v1_edge_w = self.v1_edge_layers[i](v1_edge_emb)
            v2_edge_w = self.v2_edge_layers[i](v2_edge_emb)
            # [batch_size, v1_num, v1_num, edge_head_num]
            v1_sim = torch.bmm(v1nodes, v1nodes.permute((0, 2, 1)))
            v2_sim = torch.bmm(v2nodes, v2nodes.permute((0, 2, 1)))
            # v1_sim: [batch_size, v1_num, v1_num]
            v1_alpha = (v1_sim.view(batch_size, v1_num, v1_num, 1)
                        * v1_edge_w).permute(0, 3, 1, 2)
            v2_alpha = (v2_sim.view(batch_size, v2_num, v2_num, 1)
                        * v2_edge_w).permute(0, 3, 1, 2)

            # v1_alpha: [batch_size, edge_head_num, v1_num, v1_num]
            v1_alpha = v1_alpha.reshape(
                batch_size*self.edge_head_num, v1_num, v1_num)
            v2_alpha = v2_alpha.reshape(
                batch_size*self.edge_head_num, v2_num, v2_num)
            v1_alpha = F.softmax(v1_alpha, dim=-1)
            v2_alpha = F.softmax(v2_alpha, dim=-1)

            v1nodes_ = v1nodes.view(
                1, batch_size, v1_num, -1).expand(self.edge_head_num, -1, -1, -1)
            v2nodes_ = v2nodes.view(
                1, batch_size, v2_num, -1).expand(self.edge_head_num, -1, -1, -1)
            v1nodes_ = v1nodes_.reshape((-1, v1_num, v1_dim))
            v2nodes_ = v2nodes_.reshape((-1, v2_num, v2_dim))

            v1nodes_ = torch.bmm(v1_alpha, v1nodes_).view(
                batch_size, self.edge_head_num, v1_num, v1_dim)
            v2nodes_ = torch.bmm(v2_alpha, v2nodes_).view(
                batch_size, self.edge_head_num, v2_num, v2_dim)

            v1nodes = torch.mean(v1nodes_, dim=1)
            v2nodes = torch.mean(v2nodes_, dim=1)
        v1emb = v1nodes.mean(dim=1)
        v2emb = v2nodes.mean(dim=1)
        nodes = nodes.view(batch_size, -1)

        v1emb = torch.cat([v1emb, nodes], dim=1).view((batch_size, 1, -1))
        v2emb = torch.cat([v2emb, nodes], dim=1).view((batch_size, 1, -1))
        nodes = torch.cat([nodes, nodes], dim=1).view((batch_size, 1, -1))
        v1v2n = torch.cat([v1emb, v2emb, nodes], dim=1)
        alpha = F.leaky_relu(self.mm_layer(v1v2n))
        alpha = F.softmax(alpha, dim=1)
        f2 = (v1v2n*alpha).mean(dim=1)
        cf2 = self.drop(f2)
        logits = self.classify_layer(cf2)
        return f2, logits


if __name__ == "__main__":
    v1_dim = 16
    v2_dim = 32
    v1_sample_num = 32
    v2_sample_num = 37
    hidden_dim = 24
    v1_edge_attr_dim = 1
    v2_edge_attr_dim = 1
    edge_layer_num = 2
    edge_head_num = 3
    edge_emb_dim = 8
    edge_type_num = 10
    label_num = 5
    model = HMM(v1_dim=16,
                v2_dim=32,
                v1_sample_num=v1_sample_num,
                v2_sample_num=v2_sample_num,
                hidden_dim=hidden_dim,
                v1_edge_attr_dim=v1_edge_attr_dim,
                v2_edge_attr_dim=v2_edge_attr_dim,
                edge_layer_num=edge_layer_num,
                edge_head_num=edge_head_num,
                edge_emb_dim=edge_emb_dim,
                edge_type_num=edge_type_num, label_num=label_num
                )
    batch_size = 128
    nodes = torch.randn(batch_size, v1_dim)
    v1_nodes = torch.randn(batch_size, v1_sample_num, v1_dim)
    v2_nodes = torch.randn(batch_size, v2_sample_num, v2_dim)
    v1adj = torch.randint(0, edge_type_num, size=(
        batch_size, v1_sample_num+1, v1_sample_num+1))
    v2adj = torch.randint(0, edge_type_num, size=(
        batch_size, v2_sample_num+1, v2_sample_num+1))
    v1_edge_attr = torch.randn(
        batch_size, v1_sample_num+1, v1_sample_num+1)
    v2_edge_attr = torch.zeros(
        (batch_size, v2_sample_num+1, v2_sample_num+1))
    res = model(nodes, v1_nodes, v2_nodes, v1adj,
                v2adj, v1_edge_attr, v2_edge_attr)
