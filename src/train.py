import pathlib
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from loss import SupConLoss

from models.vgg import SupConVGG, VGGLinearClassifier
from utils.tools import EarlyStopping, generatePositive, label_hard_acc


# proposed method
def train_supcon(options, train_loader, valid_loader, augmentation: dict, label_counts=None):
    """
    训练基于监督对比学习的模型
    
    Args:
        options: 包含训练配置的选项对象，包含epochs、temperature、gamma等超参数
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        augmentation: 包含数据增强策略的字典，至少有'invariant_aug'和'to_tensor'键
        label_counts: 训练集中各类别的样本数量统计
    
    Returns:
        dict: 包含训练和验证的准确率和损失历史记录
    """
    
    # 构建模型名称，例如vgg16
    model_name = "vgg" + options.model_config
    
    # 设置日志和模型保存路径
    log_path = pathlib.Path("log")
    pt_path = pathlib.Path("saved")
    
    # 修改日志目录创建部分
    logdir = (
        log_path
        / options.dataset
        / str(options.exp_id)
        / (model_name + options.head)  # 模型架构加头类型
        / str(options.set_size)
        / str(options.gamma)
        / f"adttem_{options.adaptive_temperature}"
        / f"hnm_{options.hard_negative_mining}"
        / f"center_{options.use_center_loss}"
        / f"weights_{options.use_class_weights}"
    )
    
    # 修改模型保存目录创建部分
    pt_dir = (
        pt_path
        / options.dataset
        / str(options.exp_id)
        / (model_name + options.head)  # 模型架构加头类型
        / str(options.set_size)
        / str(options.gamma)
        / f"adttem_{options.adaptive_temperature}"
        / f"hnm_{options.hard_negative_mining}"
        / f"center_{options.use_center_loss}"
        / f"weights_{options.use_class_weights}"
    )
    
    # 创建目录（如果不存在）
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=False)
    if not pt_dir.exists():
        pt_dir.mkdir(parents=True, exist_ok=False)
    
    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir=logdir)
    writer.flush()
    # 设置训练设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    EPOCHS = options.epochs
    # 设置训练集的数据增强（生成正样本对）
    train_loader.dataset.set_transform(generatePositive(augmentation["invariant_aug"]))
    # 设置验证集的数据转换（仅转为张量）
    valid_loader.dataset.set_transform(augmentation["to_tensor"])
    
    # 找到创建SupConLoss的部分，修改为：
    
    # 根据数据集类型设置损失函数和类别数
    if options.dataset == "mixedwm38":
        ce_loss = nn.BCEWithLogitsLoss()  # 多标签分类使用二元交叉熵
        num_classes = 8
        sup_con_loss = SupConLoss(
            temperature=options.temperature,
            base_temperature=options.temperature,
            adaptive_temperature=options.adaptive_temperature,
            hard_negative_mining=options.hard_negative_mining,
            negative_ratio=options.negative_ratio,
            use_center_loss=options.use_center_loss,
            center_weight=options.center_weight,
            num_classes=num_classes,
            feat_dim=128,
        )
    
    elif options.dataset == "wm811k":
        # 根据选项决定是否使用权重
        if options.use_class_weights and label_counts is not None:
            # 方法1：使用频率倒数计算权重
            total_samples = len(train_loader.dataset)
            num_classes = len(label_counts)
            # 确保所有类别都有对应的权重
            class_weights = torch.zeros(num_classes, device=device)
        
            for class_idx, count in label_counts.items():
                # 使用频率倒数计算权重
                class_weights[class_idx] = total_samples / (num_classes * count)
        
            # 可选：权重归一化
            class_weights = class_weights / class_weights.max() if class_weights.max() > 0 else class_weights
            ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # 不使用权重或无法获取标签统计
            ce_loss = nn.CrossEntropyLoss()
        
        num_classes = 9
        sup_con_loss = SupConLoss(
            temperature=options.temperature,
            base_temperature=options.temperature,
            adaptive_temperature=options.adaptive_temperature,
            hard_negative_mining=options.hard_negative_mining,
            negative_ratio=options.negative_ratio,
            use_center_loss=options.use_center_loss,
            center_weight=options.center_weight,
            num_classes=num_classes,
            feat_dim=128,
        )
    
    # 初始化对比学习模型和分类器
    model = SupConVGG(name=model_name, head=options.head, feat_dim=128).to(device)
    classifier = VGGLinearClassifier(
        encoder_name=model_name, num_classes=num_classes
    ).to(device)
    
    # 合并模型参数
    params = list(model.parameters()) + list(classifier.parameters())
    
    # 初始化Adam优化器
    optimizer = optim.Adam(params, lr=1e-5, betas=(0.9, 0.999))
    
    # 初始化早停策略
    earlystop = EarlyStopping(patience=options.patience)
    
    # 初始化损失和准确率跟踪变量
    min_val_loss = torch.inf
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    c = options.gamma  # 对比损失和分类损失之间的平衡超参数
    
    # 开始训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        train_loss = 0
        train_acc = 0
        model.train()
        classifier.train()
        num_step = 0
        
        for idx, (image, labels) in enumerate(train_loader, start=1):
            # 清零梯度
            optimizer.zero_grad()
            
            # 从数据加载器获取两个增强视图
            x1 = image[0]  # 第一个视图
            x2 = image[1]  # 第二个视图
            
            # 合并两个视图用于批处理
            images = torch.cat([x1, x2], dim=0)
            
            # 将数据移至设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播：获取特征表示和投影特征
            repr, projected = model(images)
            
            # 分割投影特征，分别对应两个增强视图
            bsz = x1.shape[0]
            h1, h2 = torch.split(projected, [bsz, bsz], dim=0)
            h = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1)
            
            # 计算对比学习损失
            contrastive_loss = sup_con_loss(features=h, labels=labels)
            
            # 计算分类损失
            logits = classifier(repr)
            label_cat = torch.cat([labels, labels], dim=0)  # 标签也需要与输入匹配
            classfication_loss = ce_loss(logits, label_cat)
            
            # 组合损失：对比损失和分类损失的加权和
            net_loss = c * contrastive_loss + classfication_loss
            
            # 反向传播
            net_loss.backward()
            
            # 根据数据集类型计算准确率
            if options.dataset == "mixedwm38":
                # 多标签分类：使用sigmoid和0.5阈值
                probs = logits.sigmoid()
                predicted = (probs > 0.5).float()
            else:
                # 单标签分类：使用softmax和argmax
                probs = logits.softmax(dim=1)
                predicted = probs.argmax(1)
            
            # 累加准确率和损失
            train_acc += label_hard_acc(predicted, label_cat)
            train_loss += net_loss.item()
            
            # 更新参数
            optimizer.step()
            num_step = idx
        
        # 计算平均训练准确率和损失
        train_acc /= num_step
        train_loss /= num_step
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 记录到TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        
        # 验证阶段
        valid_loss = 0
        valid_acc = 0
        model.eval()
        classifier.eval()
        
        num_step = 0
        with torch.no_grad():  # 验证阶段不需要计算梯度
            for idx, (image, labels) in enumerate(valid_loader, start=1):
                image = image.to(device)
                labels = labels.to(device)
                
                # 前向传播：只需要特征表示用于分类
                repr, _ = model(image)
                logits = classifier(repr)
                classfication_loss = ce_loss(logits, labels)
                
                bsz = image.shape[0]
                
                # 根据数据集类型计算准确率
                if options.dataset == "mixedwm38":
                    probs = logits.sigmoid()
                    predicted = (probs > 0.5).float()
                else:
                    probs = logits.softmax(dim=1)
                    predicted = probs.argmax(1)
                
                # 累加准确率和损失
                valid_acc += label_hard_acc(predicted, labels)
                valid_loss += classfication_loss.item()
                num_step = idx
        
        # 计算平均验证准确率和损失
        valid_loss /= num_step
        valid_acc /= num_step
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # 记录到TensorBoard
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Acc/valid", valid_acc, epoch)
        
        # 打印当前轮次的训练和验证结果
        print("Epoch -> ", epoch)
        print(" Training Loss -> ", train_loss, "Valid Loss -> ", valid_loss)
        print(" Training Acc -> ", train_acc, "Valid Acc -> ", valid_acc)
        
        # 保存最佳模型
        if min_val_loss > valid_loss:
            print("Writing Model at epoch ", epoch)
            model_pt_file = pt_dir / (model_name + "_model.pt")
            classifier_pt_file = pt_dir / (model_name + "_classifier.pt")
            torch.save(model.state_dict(), model_pt_file)
            torch.save(classifier.state_dict(), classifier_pt_file)
            min_val_loss = valid_loss
        
        # 早停检查
        earlystop(valid_loss, optimizer)
        
        if earlystop.early_stop:
            break
    
    # 汇总训练结果
    results = {
        "train_acc": train_accs,
        "train_losses": train_losses,
        "valid_accs": valid_accs,
        "valid_losses": valid_losses,
    }
    
    return results