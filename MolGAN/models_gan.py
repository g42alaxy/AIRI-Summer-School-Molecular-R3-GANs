import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import (
    GraphConvolution,
    GraphAggregation,
    MultiDenseLayer,
)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(
            -1, self.edges, self.vertexes, self.vertexes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(
        self,
        conv_dim,
        m_dim,
        b_dim,
        with_features: bool = False,
        f_dim: int = 0,
        dropout_rate: float = 0.0,
        batch_discriminator: bool = False,
    ):
        super().__init__()

        self.activation_f = torch.nn.Tanh()
        self.batch_discriminator = batch_discriminator

        graph_conv_dim, aux_dim, linear_dim = conv_dim  # unpack

        # ---------------- Encoder (shared with everything inside this block) -------------
        self.gcn_layer = GraphConvolution(
            m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate
        )
        self.agg_layer = GraphAggregation(
            graph_conv_dim[-1] + m_dim,
            aux_dim,
            self.activation_f,
            with_features,
            f_dim,
            dropout_rate,
        )
        self.multi_dense_layer = MultiDenseLayer(
            aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate
        )

        # ---------------- Batch-discriminator pathway -----------------------------------
        if self.batch_discriminator:
            bd_dim = linear_dim[-2] // 8  # same heuristic as TF implementation
            self.bd_fc1 = nn.Linear(aux_dim, bd_dim)
            self.bd_fc2 = nn.Linear(bd_dim, bd_dim)
            final_in = linear_dim[-1] + bd_dim
        else:
            self.bd_fc1 = None
            self.bd_fc2 = None
            final_in = linear_dim[-1]

        # Final scalar output
        self.output_layer = nn.Linear(final_in, 1)

    # -------------------------------------------------------------------------
    # Forward pass: returns (logits, features) where *features* is the vector
    # after the dense stack (and after minibatch concat if enabled) so that the
    # feature-matching loss can use it.
    # -------------------------------------------------------------------------
    def forward(self, adj, hidden, node, activation=None):
        # Pre-process adjacency to separate edge types
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)

        # Encoder
        h_enc = self.gcn_layer(node, adj, hidden)
        h_enc = self.agg_layer(h_enc, node, hidden)

        # Local dense stack
        h = self.multi_dense_layer(h_enc)  # (B, linear_dim[-1])

        # Minibatch discriminator (batch-level feature) if enabled
        if self.batch_discriminator:
            bd = torch.tanh(self.bd_fc1(h_enc))  # (B, bd_dim)
            bd_mean = bd.mean(dim=0, keepdim=True)  # (1, bd_dim)
            bd = torch.tanh(self.bd_fc2(bd_mean))  # (1, bd_dim)
            bd = bd.repeat(h.size(0), 1)  # tile to batch
            h = torch.cat([h, bd], dim=-1)

        # Final score
        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
