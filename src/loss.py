import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        base_temperature=0.1,
        adaptive_temperature=False,
        hard_negative_mining=False,
        negative_ratio=0.5,
        use_center_loss=False,
        center_weight=0.3,
        num_classes=9,
        feat_dim=128,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.adaptive_temperature = adaptive_temperature
        self.hard_negative_mining = hard_negative_mining
        self.negative_ratio = negative_ratio
        self.use_center_loss = use_center_loss
        self.center_weight = center_weight
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 初始化类别中心（如果使用中心损失）
        if use_center_loss:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
            nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels=None, mask=None, return_features=False):
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
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
        elif labels.shape[1] == 1:  # 2D 标签但只有一个通道
            labels = F.one_hot(labels.squeeze(1), num_classes=self.num_classes).float()

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

        # 新增：难例挖掘
        if self.hard_negative_mining:
            with torch.no_grad():
                # 非对角线元素为负样本（排除自己）
                neg_mask = (1 - mask) * logits_mask
                # 对每个样本找出最难的负样本（相似度最高的负样本）
                num_neg_samples = max(1, int(logits.shape[1] * self.negative_ratio))
                # 获取每个样本的负样本logits
                neg_logits = logits.masked_fill(mask.bool(), -float('inf'))
                # 找出最难的负样本索引
                _, hard_neg_idx = neg_logits.topk(k=num_neg_samples, dim=1, largest=True)
                # 创建难例掩码
                hard_neg_mask = torch.zeros_like(logits_mask, device=device)
                # 将对应位置设为1
                batch_idx = torch.arange(hard_neg_idx.shape[0]).unsqueeze(1).repeat(1, num_neg_samples)
                hard_neg_mask[batch_idx, hard_neg_idx] = 1
                # 应用难例掩码
                logits_mask = hard_neg_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 计算基本对比损失
        contrastive_loss = -(current_temp / self.base_temperature) * mean_log_prob_pos
        contrastive_loss = contrastive_loss.view(anchor_count, batch_size).mean()
        
        # 总损失初始化为对比损失
        total_loss = contrastive_loss
        
        # 新增：中心损失
        if self.use_center_loss:
            # 获取原始标签
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                target = torch.argmax(labels, dim=1)
            else:
                target = labels.squeeze() if len(labels.shape) > 1 else labels
                
            # 将centers移动到与features相同的设备
            centers = self.centers.to(device)
            
            # 对多视图求平均
            avg_features = features.mean(dim=1)
            
            # 计算中心损失
            center_loss = 0
            for i in range(self.num_classes):
                if (target == i).any():
                    center_i = centers[i].unsqueeze(0)
                    feat_i = avg_features[target == i]
                    center_loss += F.mse_loss(feat_i, center_i.repeat(feat_i.size(0), 1))
            
            # 组合损失
            total_loss = contrastive_loss * (1 - self.center_weight) + center_loss * self.center_weight
        
        if return_features:
            return total_loss, contrast_feature
            
        return total_loss