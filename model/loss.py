import torch
from torch.nn import functional as F

class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim=1)
        size = outputs.shape[0]
        res = torch.sum(torch.mul(outputs, targets), 1) / torch.sum(targets, 1)
        # print(res.shape, res)
        return -torch.sum(res) / size


def MyKMajorityLoss(alphas_part, k, sigma=0.02):
    size = alphas_part.shape[0]
    loss_wt = 0.0
    for i in range(size):
        tmp_list, _ = torch.sort(alphas_part[i], 1, True)
        alphas_part_sum = torch.tensor([0]).cuda()
        for j in range(alphas_part.shape[2] - k):
            alphas_part_sum = alphas_part_sum + tmp_list[0][j + k]

        loss_tmp = 0.0
        for j in range(k):
            loss_tmp += max(torch.Tensor([0]).cuda(), sigma - (tmp_list[0][j] - alphas_part_sum / (alphas_part.shape[2] - k)))
        loss_wt += loss_tmp * 1.0 / k
    return loss_wt / size


def MyRRBLoss(alphas_part, k, sigma=0.15):
    # print('type', k)
    size = alphas_part.shape[0]
    loss_wt = 0.0
    for i in range(size):
        tmp_list, _ = torch.sort(alphas_part[i], 1, True)
        alphas_part_sum = torch.tensor([0]).cuda()
        alphas_part_sum2 = torch.tensor([0]).cuda()
        for j in range(alphas_part.shape[2] - k):
            alphas_part_sum = alphas_part_sum + tmp_list[0][j + k]
        for j in range(k):
            alphas_part_sum2 = alphas_part_sum2 + tmp_list[0][j]
        loss_tmp = max(torch.Tensor([0]).cuda(), sigma - (alphas_part_sum2 / k - alphas_part_sum / (alphas_part.shape[2] - k)))
        loss_wt += loss_tmp
    return loss_wt / size