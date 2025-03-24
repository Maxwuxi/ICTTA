import os
import re
import pandas as pd

# 定义目录路径
directory = "./output_wrn_28_10"  # 当前目录，可以根据需要修改

# 获取目录下所有 .txt 文件
txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

# 初始化表格
columns = ["error"]
# 读取第一个文件以获取 TYPE 字段
if txt_files:
    with open(os.path.join(directory, txt_files[0]), "r") as f:
        content = f.readlines()
    type_line = [line for line in content if "TYPE" in line][0]
    types = re.findall(r"['\w_]+", type_line)[1:]  # 提取 TYPE 后面的字段
    columns.extend(types)
columns.append("mean")

precision_table = pd.DataFrame(columns=columns)
f1_table = pd.DataFrame(columns=columns)
recall_table = pd.DataFrame(columns=columns)
error_table = pd.DataFrame(columns=columns)

# 定义一个函数来提取所有指标
def extract_all_metrics(content):
    error_pattern = re.compile(rf"error %.*?(\d+\.\d+)%")
    precision_pattern = re.compile(rf"Precision: (\d+\.\d+)%")
    recall_pattern = re.compile(rf"Recall: (\d+\.\d+)%")
    f1_pattern = re.compile(rf"F1: (\d+\.\d+)%")
    
    error_matches = error_pattern.findall(content)
    precision_matches = precision_pattern.findall(content)
    recall_matches = recall_pattern.findall(content)
    f1_matches = f1_pattern.findall(content)
    
    return (
        [float(val) for val in error_matches],
        [float(val) for val in precision_matches],
        [float(val) for val in recall_matches],
        [float(val) for val in f1_matches]
    )

# 读取每个文件并提取数据
for file in txt_files:
    with open(os.path.join(directory, file), "r") as f:
        content = f.read()
    
    # 提取文件名中的方法名称
    method = os.path.splitext(file)[0]
    
    # 初始化行数据
    precision_row = {"precision": method}
    f1_row = {"f1": method}
    recall_row = {"recall": method}
    error_row = {"error": method}
    error, precision, recall, f1 = extract_all_metrics(content)
    i=0
    # 提取每个类型的指标
    for t in types:
        
        if error:
            error_row[t] = error[i]
            precision_row[t] = precision[i]
        if recall:
            recall_row[t] = recall[i]
        if f1:
            f1_row[t] = f1[i]
        i=i+1
    
    # 添加到表格
    precision_table = precision_table.append(precision_row, ignore_index=True)
    f1_table = f1_table.append(f1_row, ignore_index=True)
    recall_table = recall_table.append(recall_row, ignore_index=True)
    error_table = error_table.append(error_row, ignore_index=True)

# 计算平均值
precision_table["mean"] = precision_table[types].mean(axis=1)
f1_table["mean"] = f1_table[types].mean(axis=1)
recall_table["mean"] = recall_table[types].mean(axis=1)
error_table["mean"] = error_table[types].mean(axis=1)

# 保存到CSV文件
output_file = "output_wrn_28_10"+".csv"
with open(output_file, "w") as f:
    f.write("\n\nError Table:\n")
    f.write(error_table.to_csv(index=False))
    f.write("Precision Table:\n")
    f.write(precision_table.to_csv(index=False))
    f.write("\n\nRecall Table:\n")
    f.write(recall_table.to_csv(index=False))
    f.write("\n\nF1 Table:\n")
    f.write(f1_table.to_csv(index=False))



print(f"Tables have been saved to {output_file}")