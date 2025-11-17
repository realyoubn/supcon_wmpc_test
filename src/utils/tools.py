import numpy as np
import pathlib
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
)
import json


class generatePositive:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return [x_1, x_2]


class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.lr = None

    def __call__(self, val_loss, optimizer):

        score = val_loss
        new_lr = float(optimizer.param_groups[0]["lr"])
        if self.lr is None:
            self.lr = new_lr
        else:
            if self.lr != new_lr:
                self.counter = 0
                self.lr = new_lr

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def evaluation(y_gt, y_hat, label_counts):
    y_gt = y_gt.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()

    labels = label_counts.index
    cfm = confusion_matrix(y_gt, y_hat)
    macro_f1 = f1_score(y_gt, y_hat, labels=labels, average="macro")
    return macro_f1, cfm


def parse_result(options, result_dict, train_infer: str):
    
    # 使用'vgg'作为默认模型名称，因为model_config选项对应VGG深度
    model_name = "vgg" + options.model_config
    
    # 修改结果保存路径，与pt_dir格式保持一致
    result_path = pathlib.Path("results")
    
    result_dir = (
        result_path
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
    
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=False)
    
    # 文件名中包含train_infer标识，区分训练和推理结果
    result_json = result_dir / f"results_{train_infer}.json"
    
    # 确保结果可序列化
    serializable_results = {}
    for key, value in result_dict.items():
        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            # 处理numpy数组
            serializable_results[key] = [v.tolist() for v in value]
        elif isinstance(value, np.ndarray):
            # 单个numpy数组
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(result_json, "w") as fjson:
        json.dump(serializable_results, fjson, indent=4)
    
    print(f"Results saved to {result_json}")


def calculate_metrics(options, pred, target, label_index=None):
    pred = pred.detach().cpu().numpy().astype(int)
    target = target.detach().cpu().numpy().astype(int)

    if options.dataset == "mixedwm38":
        gt_none = (target.sum(axis=1) == 0).astype(int).reshape(1, -1).T
        pred_none = (pred.sum(axis=1) == 0).astype(int).reshape(1, -1).T

        pred = np.concatenate([pred, pred_none], axis=1)
        target = np.concatenate([target, gt_none], axis=1)

    return {
        "label_f1": f1_score(
            y_true=target, y_pred=pred, average=None, zero_division=0
        ).tolist(),
        "micro/f1": f1_score(
            y_true=target,
            y_pred=pred,
            average="micro",
            zero_division=0,
            labels=label_index,
        ),
        "macro/f1": f1_score(
            y_true=target,
            y_pred=pred,
            average="macro",
            zero_division=0,
            labels=label_index,
        ),
    }


def label_hard_acc(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return accuracy_score(y_true=target, y_pred=pred)