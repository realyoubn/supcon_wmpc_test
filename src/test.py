import torch
import pathlib
from models.vgg import SupConVGG, VGGLinearClassifier
from utils.tools import calculate_metrics

def test_supcon(options, test_loader, label_counts, augmentation: dict):
    """
    测试基于对比学习训练的模型性能
    
    参数:
        options: 包含各种配置选项的对象，如模型配置、数据集、实验ID等
        test_loader: 测试数据加载器，提供批量测试数据
        label_counts: 标签计数信息，用于WM811k数据集的指标计算
        augmentation: 数据增强字典，包含to_tensor转换方法
        
    返回:
        results: 包含测试性能指标的字典
    """
    # 构建模型名称，如"vgg16"、"vgg19"等
    model_name = "vgg" + options.model_config
    
    # 设置模型权重保存路径的根目录
    pt_path = pathlib.Path("saved")
    
    # 构建完整的模型权重路径，根据数据集、实验ID、模型配置等动态生成
    pt_dir = (
        pt_path
        / options.dataset         # 数据集名称(mixedwm38或wm811k)
        / str(options.exp_id)     # 实验ID
        / (model_name + options.head)  # 模型名称+投影头类型
        / str(options.set_size)   # 训练集大小
        / str(options.gamma)      # 损失平衡参数
        / str(options.adaptive_temperature)  # 自适应温度参数
    )
    
    # 设置运行设备(GPU优先)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义模型和分类器的权重文件路径
    model_pt_file = pt_dir / (model_name + "_model.pt")         # 特征提取器权重文件
    classifier_pt_file = pt_dir / (model_name + "_classifier.pt")  # 分类器权重文件
    
    # 根据数据集类型设置类别数和标签索引
    if options.dataset == "mixedwm38":
        numclasses = 8        # MixedWM38数据集有8个类别
        label_index = None    # 多标签分类不需要标签索引
    elif options.dataset == "wm811k":
        numclasses = 9        # WM811k数据集有9个类别
        label_index = label_counts.index  # 获取标签索引用于后续指标计算
    
    # 初始化特征提取器模型
    model = SupConVGG(name=model_name, head=options.head, feat_dim=128).to(device)
    
    # 初始化线性分类器
    classifier = VGGLinearClassifier(
        encoder_name=model_name, num_classes=numclasses
    ).to(device)
    
    # 设置测试数据集的转换方法为to_tensor(不使用数据增强)
    test_loader.dataset.set_transform(augmentation["to_tensor"])
    
    # 加载训练好的特征提取器权重
    model.load_state_dict(torch.load(model_pt_file))
    model = model.to(device)
    
    # 加载训练好的分类器权重
    classifier.load_state_dict(torch.load(classifier_pt_file))
    classifier = classifier.to(device)
    
    # 初始化用于存储真实标签和预测标签的张量
    y_gt = torch.Tensor().to(device)  # 真实标签
    y_hat = torch.Tensor().to(device)  # 预测标签
    
    # 设置模型为评估模式(禁用dropout等训练时特有操作)
    model.eval()
    classifier.eval()
    
    # 不计算梯度，节省内存并加速推理
    with torch.no_grad():
        # 遍历测试数据集
        for idx, (image, target) in enumerate(test_loader):
            # 将数据移至指定设备
            image = image.to(device)
            target = target.to(device)
            
            # 通过特征提取器获取特征表示
            # _表示投影特征，测试阶段不需要使用
            repr, _ = model(image)
            
            # 将特征输入分类器得到预测结果
            output = classifier(repr)
            
            # 根据数据集类型进行不同的预测后处理
            if options.dataset == "mixedwm38":
                # 多标签分类：应用sigmoid激活并使用0.5阈值
                probs = output.sigmoid()
                predicted = (probs > 0.5).float()
            else:
                # 单标签分类：应用softmax并取概率最大的类别
                probs = output.softmax(dim=1)
                predicted = probs.argmax(1)
            
            # 累积真实标签和预测标签
            y_gt = torch.cat((y_gt, target), 0)
            y_hat = torch.cat((y_hat, predicted), 0)
    
    # 计算并返回评估指标
    results = calculate_metrics(
        options=options, pred=y_hat, target=y_gt, label_index=label_index
    )
    
    # 清理GPU缓存，释放内存
    torch.cuda.empty_cache()
    
    return results  # 返回包含准确率、F1等评估指标的结果字典