import os
import shutil
from pathlib import Path
import pandas as pd

source_dir = "F:/Data/Sewer"
destination_dir = "F:/Data/New"
csv_file = os.path.join(source_dir, "SewerML_Train.csv")
train_dirs = [f"train{i:02d}" for i in range(14)]

# 读取csv文件
df = pd.read_csv(csv_file)

# 忽略 1, 2, 20 列
labels_to_ignore = [1, 2, 20, 21]
df = df.drop(columns=df.columns[labels_to_ignore])

# 创建目标目录
Path(destination_dir).mkdir(parents=True, exist_ok=True)

# 创建子文件夹
for col in df.columns[1:]:
    Path(os.path.join(destination_dir, col)).mkdir(parents=True, exist_ok=True)

# 复制图片到相应的分类子文件夹
counter = dict(zip(df.columns[1:], [0] * len(df.columns[1:])))

for index, row in df.iterrows():
    file_moves = []
    for col in df.columns[1:]:
        if row[col] == 1:
            file_moves.append(col)

    if len(file_moves) == 1:
        src_file = None
        for train_dir in train_dirs:
            full_path = os.path.join(source_dir, train_dir, row[0])
            if os.path.exists(full_path):
                src_file = full_path
                break

        if src_file is not None:
            dest_file = os.path.join(destination_dir, file_moves[0], row[0])
            shutil.copyfile(src_file, dest_file)
            counter[file_moves[0]] += 1

            # 检查是否已达到每个类别500张图片的要求
            if all(count >= 500 for count in counter.values()):
                break
#%%
