import torch
import torch.nn as nn
from .utils import custom_global_avg_pool


class CFD_DC(nn.Module):
    def __init__(self, num_classes, num_nodes, num_features_per_view, fea_out, d_prob, reduction=2):

        super(CFD_DC, self).__init__()
        self.num_nodes = num_nodes
        self.num_features_per_view = num_features_per_view
        self.fea_out = fea_out  # compressed dimension

        self.compressors = nn.ModuleList()
        for i in range(num_nodes):
            compressor = nn.Sequential(
                nn.Linear(num_features_per_view[i], int(num_features_per_view[i] / 4)),
                nn.BatchNorm1d(int(num_features_per_view[i] / 4)), nn.ReLU(inplace=True), nn.Dropout(d_prob),
                nn.Linear(int(num_features_per_view[i] / 4), fea_out), nn.BatchNorm1d(fea_out), nn.ReLU(inplace=True)
            )
            self.compressors.append(compressor)

        self.classifiers = nn.ModuleList()
        for i in range(num_nodes):
            # Each classifier receives: local features + (N-1) * compressed features
            classifier = nn.Sequential(
                nn.Linear(num_features_per_view[i] + (num_nodes - 1) * fea_out, num_classes),
                nn.BatchNorm1d(num_classes)
            )
            self.classifiers.append(classifier)

        self.se_layers = nn.ModuleList()
        for i in range(num_nodes):
            self.se_layers.append(SELayer(num_nodes, reduction=reduction))

    def forward(self, input_features, node_failure_mask=None):

        # If a node is faulty, its local feature is zeroed out before compression
        if node_failure_mask is not None:
            for i in range(self.num_nodes):
                # [batch_size, 1]
                mask = node_failure_mask[:, i].unsqueeze(1)
                # Broadcast mask to [batch_size, feature_dim_i] and zero out
                input_features[i] = input_features[i] * (~mask)

        compressed_Fea_list = []
        for i, input_item in enumerate(input_features):
            input_item = input_item.to(next(self.parameters()).device, dtype=torch.float32)
            # Even for a faulty node (all-zero input), the compressor will run, producing g_i,k (possibly a zero vector)
            compressed_features = self.compressors[i](input_item)
            compressed_Fea_list.append(compressed_features)

        all_node_logits = []
        for i in range(self.num_nodes):  # Perform local classification for each node i

            # Collect compressed features from other nodes j!=i
            other_node_compressed_feature = []
            for j in range(self.num_nodes):
                if j != i:
                    other_node_compressed_feature.append(compressed_Fea_list[j])

            current_local_features = input_features[i].to(next(self.parameters()).device, dtype=torch.float32)
            current_allother_features = torch.stack(other_node_compressed_feature, dim=1)

            features_list = [current_local_features] + [current_allother_features[:, k] for k in
                                                        range(self.num_nodes - 1)]

            currentnode_avg_pool = custom_global_avg_pool(features_list)

            current_allnode_weight = self.se_layers[i](currentnode_avg_pool.to(next(self.parameters()).device))

            weighted_features = [features_list[uu] * current_allnode_weight[:, uu:uu + 1]
                                 for uu in range(self.num_nodes)]

            weighted_all_feature = torch.cat(weighted_features, dim=1)

            current_node_logits = self.classifiers[i](weighted_all_feature)
            all_node_logits.append(current_node_logits)

        all_node_logits = torch.stack(all_node_logits, dim=1)
        # sum_logits = torch.sum(all_node_logits, dim=1)

        return all_node_logits  # [batch_size, num_nodes, num_classes]


class SELayer(nn.Module):
    def __init__(self, num_nodes, reduction=2):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_nodes, num_nodes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_nodes // reduction, num_nodes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, avg_pool):
        y = self.fc(avg_pool)
        return y


class CFD_DC_NoWeighting(nn.Module):
    def __init__(self, num_classes, num_nodes, num_features_per_view, fea_out, d_prob):
        super(CFD_DC_NoWeighting, self).__init__()

        self.compressors = nn.ModuleList()
        for i in range(len(num_features_per_view)):
            compressor = nn.Sequential(
                nn.Linear(num_features_per_view[i], int(num_features_per_view[i] / 4)),
                nn.BatchNorm1d(int(num_features_per_view[i] / 4)), nn.ReLU(inplace=True), nn.Dropout(d_prob),
                nn.Linear(int(num_features_per_view[i] / 4), fea_out), nn.BatchNorm1d(fea_out), nn.ReLU(inplace=True)
            )
            self.compressors.append(compressor)

        self.classifiers = nn.ModuleList()
        for i in range(len(num_features_per_view)):
            classifier = nn.Sequential(
                nn.Linear(num_features_per_view[i] + (num_nodes - 1) * fea_out, num_classes),
                nn.BatchNorm1d(num_classes)
            )
            self.classifiers.append(classifier)

    def forward(self, input_features, node_failure_mask=None):
        num_nodes = len(input_features)

        # Simulate node failure
        if node_failure_mask is not None:
            for i in range(num_nodes):
                mask = node_failure_mask[:, i].unsqueeze(1)
                input_features[i] = input_features[i] * (~mask)

        compressed_Fea_list = []
        for i, input_item in enumerate(input_features):
            input_item = input_item.to(next(self.parameters()).device, dtype=torch.float32)
            compressed_features = self.compressors[i](input_item)
            compressed_Fea_list.append(compressed_features)

        all_node_logits = []
        for i in range(num_nodes):
            other_node_compressed_feature = []
            for j in range(num_nodes):
                if j != i:
                    other_node_compressed_feature.append(compressed_Fea_list[j])

            current_local_features = input_features[i].to(next(self.parameters()).device, dtype=torch.float32)
            current_allother_features = torch.stack(other_node_compressed_feature, dim=1)
            num_sample = current_allother_features.shape[0]

            current_all_features = torch.cat([current_local_features, current_allother_features.view(num_sample, -1)],
                                             dim=1)

            current_node_logits = self.classifiers[i](current_all_features)
            all_node_logits.append(current_node_logits)

        all_node_logits = torch.stack(all_node_logits, dim=1)
        # sum_logits = torch.sum(all_node_logits, dim=1)

        return all_node_logits


class CFD_DC_50Compress(nn.Module):
    def __init__(self, num_classes, num_nodes, num_features_per_view, d_prob):
        super(CFD_DC_50Compress, self).__init__()

        self.compressors = nn.ModuleList()
        self.compressed_dimensions = []

        for features in num_features_per_view:
            feature_in_dim = int(features)
            final_output_dim = int(features / 2)  # d_g = d_f / 2
            self.compressed_dimensions.append(final_output_dim)
            compressor = nn.Sequential(
                nn.Linear(feature_in_dim, final_output_dim),
                nn.BatchNorm1d(final_output_dim), nn.ReLU(inplace=True)
            )
            self.compressors.append(compressor)

        total_compressed_output = sum(self.compressed_dimensions)

        self.classifiers = nn.ModuleList()
        for i, features in enumerate(num_features_per_view):
            input_dim = features + (total_compressed_output - self.compressed_dimensions[i])
            classifier = nn.Sequential(
                nn.Linear(input_dim, int(input_dim / 8)),
                nn.BatchNorm1d(int(input_dim / 8)), nn.ReLU(inplace=True), nn.Dropout(d_prob),
                nn.Linear(int(input_dim / 8), num_classes), nn.BatchNorm1d(num_classes)
            )
            self.classifiers.append(classifier)

    def forward(self, input_features, node_failure_mask=None):
        num_nodes = len(input_features)

        # 模拟节点故障
        if node_failure_mask is not None:
            for i in range(num_nodes):
                mask = node_failure_mask[:, i].unsqueeze(1)
                input_features[i] = input_features[i] * (~mask)

        compressed_Fea_list = []
        for i, input_item in enumerate(input_features):
            input_item = input_item.to(next(self.parameters()).device, dtype=torch.float32)
            compressed_features = self.compressors[i](input_item)
            compressed_Fea_list.append(compressed_features)

        all_node_logits = []
        for i in range(num_nodes):
            other_node_compressed_feature = [compressed_Fea_list[j] for j in range(num_nodes) if j != i]
            current_local_features = input_features[i].to(next(self.parameters()).device, dtype=torch.float32)

            current_allother_features = torch.cat(other_node_compressed_feature, dim=1)
            current_all_features = torch.cat([current_local_features, current_allother_features], dim=1)

            current_node_logits = self.classifiers[i](current_all_features)
            all_node_logits.append(current_node_logits)

        all_node_logits = torch.stack(all_node_logits, dim=1)
        # sum_logits = torch.sum(all_node_logits, dim=1)

        return all_node_logits
