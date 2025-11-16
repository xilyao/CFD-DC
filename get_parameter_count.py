import torch
from torchinfo import summary
from cfd_dc.models import CFD_DC # (假设你这样命名了)

# 1. 初始化一个水下声学模型的例子
num_nodes = 10
d_feature = 1024
d_compress = 32
num_classes = 5
num_features_per_view = [d_feature] * num_nodes

model = CFD_DC(num_classes, num_nodes, num_features_per_view, d_compress, d_prob=0.7)

# 2. 打印摘要
# 这将自动计算并显示所有子模块的参数量
summary(model)