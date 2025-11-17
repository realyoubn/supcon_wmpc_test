import argparse


def parse_options():
    parser = argparse.ArgumentParser("Arguments for training")

    parser.add_argument(
        "--dataset",
        type=str,
        choices={"wm811k", "mixedwm38"},
    )

    parser.add_argument("--batch_size", type=int, help="batch size")

    parser.add_argument(
        "--set_size",
        type=int,
        help="data subset size : how many data instances you will use to train model",
    )

    parser.add_argument("--epochs", type=int, default=500, help="epochs")

    parser.add_argument("--patience", type=int, default=50, help="epochs")

    parser.add_argument(
        "--adaptive_temperature",
        action="store_true",
        help="use adaptive temperature for contrastive loss",
    )
                    
    parser.add_argument(
        "--model_config",
        type=str,
        default="16",
        choices={"11", "13", "16", "19"},
        help="model configuration(depth)\n \
            vgg = [11,13,16,19]\n",
    )

    parser.add_argument(
        "--head",
        type=str,
        default="linear",
        choices={"linear", "mlp"},
        help="Architectu1re of projection head: mlp or linear, default = linear",
    )

    parser.add_argument(
        "--gamma", default=0.1, type=float, help="gamma value for contrastive loss"
    )

    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="temperature parameter for contrastive loss",
    )

    parser.add_argument("--exp_id", type=int, help="experiment id")
    
    # 新增：难例挖掘参数
    parser.add_argument(
        "--hard_negative_mining",
        action="store_true",
        help="enable hard negative mining",
    )
    
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=0.5,
        help="ratio of hard negative samples to consider",
    )
    
    # 新增：中心对比损失参数
    parser.add_argument(
        "--use_center_loss",
        action="store_true",
        help="enable center loss",
    )
    
    parser.add_argument(
        "--center_weight",
        type=float,
        default=0.3,
        help="weight for center loss",
    )
    
    # 新增：类别权重控制参数
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="use class weights for cross entropy loss",
    )

    opt = parser.parse_args()

    return opt