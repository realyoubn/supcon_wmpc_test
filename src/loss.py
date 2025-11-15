import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        base_temperature=0.1,
        adaptive_temperature=False,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.adaptive_temperature = adaptive_temperature
        self.epoch = 0 
    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # 修复：处理 1D 和 2D 标签
        if len(labels.shape) == 1:  # 1D 标签 (类别索引)
            labels = F.one_hot(labels, num_classes=9).float()
        elif labels.shape[1] == 1:  # 2D 标签但只有一个通道
            labels = F.one_hot(labels.squeeze(1), num_classes=9).float()

        # Proposed method - weighting based on jaccard similarity
        ones = torch.ones_like(labels)
        mask = torch.matmul(labels, labels.T) / (
            torch.matmul(labels, ones.T)
            + torch.matmul(ones, labels.T)
            - torch.matmul(labels, labels.T)
        )
        mask = torch.nan_to_num(mask, nan=1.0)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # 新增：基于批内相似度动态调整温度
        if self.adaptive_temperature:
            with torch.no_grad():  # 温度调整不参与梯度计算
                # 计算所有样本对的相似度点积的中位数
                sim_matrix = torch.matmul(anchor_feature, contrast_feature.T)
                temp = torch.median(sim_matrix)  # 批内相似度中位数
                # 动态调整温度：相似度高则升温（降低区分度），相似度低则降温（增强区分度）
                adaptive_temp = max(self.temperature * (1 + temp), self.temperature * 0.5)
            current_temp = adaptive_temp
        else:
            current_temp = self.temperature  # 使用固定温度

        # 用调整后的温度计算点积
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), current_temp
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(current_temp / self.base_temperature) * mean_log_prob_pos
        
        loss = loss.view(anchor_count, batch_size)

        return loss.mean()