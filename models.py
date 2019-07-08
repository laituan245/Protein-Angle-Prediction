import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM_BRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, fc_dims, output_dim):
        super(LSTM_BRNN, self).__init__()

        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=True)

        self.fc_layers = []
        prev_dim = 2 * hidden_dim
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            self.fc_layers.append(nn.Dropout(0.8))
            self.fc_layers.append(nn.ReLU(True))
            prev_dim = fc_dim
        self.fc_layer_module = nn.ModuleList(self.fc_layers)

        self.last_fc = nn.Linear(prev_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_lengths):
        # x has shape [batch_size, max_len, input_dim]
        # x_lengths has shape [batch_size,]

        # BiLSTM Layer
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
        bilstm_output, _ = self.bilstm(packed_sequence)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(bilstm_output, batch_first=True)

        # FC Layers
        for layer in self.fc_layer_module:
            out = layer(out)

        # Last FC Layer
        out = self.last_fc(out)
        out = self.sigmoid(out) # each dimension's range will be between 0 and 1

        return out

# ResidualBlock1D will be used by Residual_CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels:
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.in_channels != self.out_channels:
            residual = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        out += residual
        out = self.relu2(out)
        return out

class Residual_CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Residual_CNN, self).__init__()

        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(ResidualBlock(in_channels=input_dim, out_channels=64))
        self.residual_blocks.append(ResidualBlock(in_channels=64, out_channels=64))
        self.residual_blocks.append(ResidualBlock(in_channels=64, out_channels=64))
        self.residual_blocks.append(ResidualBlock(in_channels=64, out_channels=128))
        self.residual_blocks.append(ResidualBlock(in_channels=128, out_channels=128))

        self.fc_1 = nn.Linear(128, 256)
        self.fc_2 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # x has shape [batch_size, max_len, input_dim]
        # lengths is not used
        x = x.permute([0, 2, 1])

        for layer in self.residual_blocks:
            x = layer(x)
        outs = x.permute([0, 2, 1])

        outs = F.relu(self.fc_1(outs))
        outs = self.fc_2(outs)
        outs = self.sigmoid(outs)

        return outs

# ResidualBlock2D
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(ResidualBlock2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels:
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = residual.permute([0, 2, 3, 1])
            residual = self.linear(residual)
            residual = residual.permute([0, 3, 1, 2])

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)
        return out

# Non-local Neural Networks
class NonLocalBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, fn_type,
                 attention_clipping=True, clipping_k=30):
        super(NonLocalBlock, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim # Embeddings for pairwise comparision
        self.fn_type = fn_type
        self.attention_clipping = attention_clipping
        self.clipping_k = clipping_k
        assert(fn_type in ['gaussian', 'embedded_gaussian', 'dot_product'])

        self.w_g = nn.Linear(input_dim, hidden_dim)
        self.w_z = nn.Linear(hidden_dim, input_dim)
        if fn_type in ['embedded_gaussian', 'dot_product']:
            self.transform_1 = nn.Linear(input_dim, embedding_dim)
            self.transform_2 = nn.Linear(input_dim, embedding_dim)

    def forward(self, x, x_lengths):
        # x must have shape [batch_size, seq_len, input_dim]
        # x_lengths has shape [batch_size]
        # output must have shape [batch_size, seq_len, input_dim]
        x_prime = self.w_g(x)

        if self.fn_type in ['embedded_gaussian', 'dot_product']:
            x_1 = self.transform_1(x)
            x_2 = self.transform_2(x)
        else:
            x_1 = x_2 = x

        # Calculate the interaction matrix
        if self.fn_type == 'gaussian' or self.fn_type == 'embedded_gaussian':
            interactions = torch.matmul(x_1, x_2.permute([0, 2, 1]))
        elif self.fn_type == 'dot_product':
            interactions = torch.matmul(x_1, x_2.permute([0, 2, 1]))

        # Create length masks
        masks = []
        max_len = max(x_lengths)
        for length in x_lengths:
            mask = [[1]] * length + [[0]] * (max_len - length)
            masks.append(mask)
        masks = torch.FloatTensor(masks)

        # Create prob matrix
        if self.fn_type == 'gaussian' or self.fn_type == 'embedded_gaussian':
            interactions =  torch.softmax(interactions, dim=-1)
        elif self.fn_type == 'dot_product':
            interactions = interactions / torch.sum(interactions, dim=-1, keepdim=True)
        interactions = interactions * masks.permute([0, 2, 1])
        interactions = interactions / torch.sum(interactions, dim=-1, keepdim=True)
        if self.attention_clipping:
            topk, indices = torch.topk(interactions, min(self.clipping_k,max_len))
            res = torch.FloatTensor(interactions.size()).zero_()
            interactions = res.scatter(-1, indices, topk)
            interactions = interactions / torch.sum(interactions, dim=-1, keepdim=True)
        #
        y = torch.matmul(interactions, x_prime)
        y = self.w_z(y)

        return x + y

# Model with Attention
class Attention_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention_Model, self).__init__()

        #
        self.residual_blocks_1 = nn.ModuleList([])
        self.residual_blocks_1.append(ResidualBlock(in_channels=input_dim, out_channels=64))
        self.residual_blocks_1.append(ResidualBlock(in_channels=64, out_channels=128))
        self.residual_blocks_1.append(ResidualBlock(in_channels=128, out_channels=256))

        #
        self.residual_blocks_2 = nn.ModuleList([])
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=1, out_channels=64))
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=64, out_channels=64))
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=64, out_channels=64))
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=64, out_channels=128))
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=128, out_channels=128))
        self.residual_blocks_2.append(ResidualBlock2D(in_channels=128, out_channels=1))

        #
        self.fc_1 = nn.Linear(512, 256)
        self.fc_2 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, _2d_map, _2d_mask):
        # Extract features from x
        x = x.permute([0, 2, 1])
        for layer in self.residual_blocks_1:
            x = layer(x)
        x = x.permute([0, 2, 1])

        # Extract features from _2d_map
        _2d_map = _2d_map.unsqueeze(1)
        for layer in self.residual_blocks_2:
            _2d_map = layer(_2d_map)
        _2d_map = _2d_map.squeeze()

        # Attention mechanism
        _2d_mask = _2d_mask.unsqueeze(1)
        _2d_map = torch.softmax(_2d_map, dim=-1)
        _2d_map = _2d_map * _2d_mask
        _2d_map = _2d_map / torch.sum(_2d_map, dim=-1, keepdim=True)

        attention_vectors = torch.matmul(_2d_map, x)

        # Concatenation
        feature_vectors = torch.cat([x, attention_vectors], dim=-1)

        # Final Output
        outs = F.relu(self.fc_1(feature_vectors))
        outs = self.fc_2(outs)
        outs = self.sigmoid(outs)

        return outs
