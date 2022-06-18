# @Date    : 2022/3/26
# @Author  : CAgAG
# @Version : 1.0
# @Function:

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')