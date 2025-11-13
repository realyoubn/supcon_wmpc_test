import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from skimage.transform import resize
DIM = 64
raw_pkl_path = Path("./dataset/LSWMD.pkl")
save_path = Path("./dataset/WM811k")

if not save_path.exists():
    save_path.mkdir()

# 加载数据框
df = pd.read_pickle(raw_pkl_path)

# 标签提取和数据预处理
f_squeeze = lambda x: str(np.squeeze(x))
df["failureType"] = df["failureType"].map(f_squeeze)
df["trianTestLabel"] = df["trianTestLabel"].map(f_squeeze)

# 移除没有指定标签的行数据
eliminate_list = ['[]']
df_with_label = df.query(f"failureType not in {eliminate_list}")
print(f"{len(df_with_label)}")
print("Label list: ", df_with_label["failureType"].unique())

# 数据清洗和标签映射
df = df.drop(['waferIndex', 'trianTestLabel', 'lotName'], axis=1)
df['failureNum']=df.failureType
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
df=df.replace({'failureNum':mapping_type})
print(df.head(10))

# 只使用有标签的晶圆图
df_withlabel = df

# 移除die size小于100的晶圆
df_withlabel = df_withlabel.drop(df_withlabel[df_withlabel['dieSize']<100].index.tolist()).reset_index()

# 初始化保存路径
labelednpyPath = []
labeled_total = len(df_withlabel)
npy_root = save_path/"wafermaps"
if not npy_root.exists():
    npy_root.mkdir()

# 分批处理图像，每次处理一个图像
for i in range(labeled_total):
    # 逐个处理每个晶圆图，而不是一次性处理所有
    wafer_map = df_withlabel.waferMap.iloc[i]
       
    # 二值化处理
    x_binary = np.where(wafer_map <= 1, 0, 1)
        
    # 调整大小
    x_resized = resize(x_binary, (DIM, DIM), order=1, preserve_range=True, anti_aliasing=False)
        
    # 重塑和格式转换
    x_resized = (x_resized.reshape(DIM, DIM, 1) * 255).astype(np.uint8)
        
    # 保存为npy文件
    fname = str(df_withlabel['index'].iloc[i])
    np.save(npy_root/fname, x_resized)
    labelednpyPath.append((npy_root/fname).name + '.npy')
        
     # 定期清理变量以释放内存
    del wafer_map, x_binary, x_resized
        
        # 显示进度
    if i % 1000 == 0:
        print('{}/{} done'.format(i, labeled_total))
            
    # 添加文件路径信息到数据框
df_withlabel['npyPath'] = labelednpyPath

# 删除不需要的列
df_withlabel.drop(['index','waferMap','dieSize','failureType'], axis=1, inplace=True)

# 保存数据框为CSV
csv_root = save_path
df_withlabel.to_csv(save_path/'labeled.csv', index=False)

# 验证数据框是否正确保存
df_withlabel = pd.read_csv('./dataset/WM811k/labeled.csv')

# 划分训练集和验证集
df_labeled_validation = df_withlabel.sample(frac=0.2)
df_withlabel.drop(df_labeled_validation.index, axis=0, inplace=True)
df_labeled_validation.to_csv(csv_root/'labeled_validation.csv', index=False)
df_withlabel.to_csv(csv_root/'labeled_training.csv', index=False)

print("数据预处理完成！")