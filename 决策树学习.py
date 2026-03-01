# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 13:45:04 2026

@author: 徐逸东
"""
# 导入pandas库
from anytree import Node, RenderTree
import pandas as pd
import math

# 指定完整的文件路径（根据您的实际路径修改）
file_path = "D:\浏览器EDGE下载\iris.data.csv"
df = pd.read_csv(file_path, header=None)

# 添加列名，使数据更清晰
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# 创建索引数组
indices = df.index  # 0, 1, 2, ..., 149
train_indices = [i for i in indices if i % 5 in [1, 2, 3, 4]]
test_indices = [i for i in indices if i % 5 == 0]

# 根据索引选择训练和测试数据
train_data = df.iloc[train_indices]
test_data = df.iloc[test_indices]

# 使用二维列表记录每个节点的父节点和拆分点,如果无父节点记作None
node_info = []

# 计算plogp,特别要定义p = 0的时候函数值取0
def plogp(pr):
    if pr <= 0:
        return 0
    else:
        return pr * math.log(pr, 2)

# 计算训练集的熵
def Ent(train_data):
    count_setosa = (train_data['species'] == 'Iris-setosa').sum()
    count_versicolor = (train_data['species'] == 'Iris-versicolor').sum()
    count_virginica = (train_data['species'] == 'Iris-virginica').sum()
    count = len(train_data)
    pr_setosa = count_setosa / count
    pr_versicolor = count_versicolor / count
    pr_virginica = count_virginica / count
    entropy = - plogp(pr_setosa) - plogp(pr_versicolor) - plogp(pr_virginica)
    return entropy

# 计算特征和划分点
def Division(train_data):
    #初始化要找的特征,划分点,以及判定标准：最小化条件熵(100足够大)
    correct_feature = None
    correct_Division_point = None
    min_condition_Ent = 100 
    for feature in train_data.columns[:-1]:
        train_data_sorted = train_data.sort_values(by=feature)
        feature_data = train_data_sorted[feature]
        # 根据每项数据均为1位小数,每次划分点增加0.1
        min_feature_data = min(feature_data)
        max_feature_data = max(feature_data)
        Division_point = min_feature_data
        while Division_point < max_feature_data:
            Division_point += 0.1
            # 拆分小于拆分点的数据和大于拆分点的数据
            smaller_data = train_data[train_data[feature] < Division_point]
            greater_data = train_data[train_data[feature] >= Division_point]
            # 计算相对熵
            smaller_data_Ent = Ent(smaller_data)
            greater_data_Ent = Ent(greater_data)
            condition_Ent = (len(smaller_data)/len(train_data) * smaller_data_Ent
                             + len(greater_data)/len(train_data) * greater_data_Ent)
            # 与之前找到最小条件熵作比较
            if condition_Ent < min_condition_Ent:
                min_condition_Ent = condition_Ent
                correct_feature = feature
                correct_Division_point = Division_point
    return correct_feature, correct_Division_point
                                 
# 分治训练
def train(train_data, decision_parent = None):
    # 判断训练集的类别是否相同
    n_categories = train_data['species'].nunique()
    if n_categories > 1:
        feature, Division_point = Division(train_data)
        node_info.append([decision_parent, [feature, 1, Division_point]])
        node_info.append([decision_parent, [feature, 0, Division_point]])
        # 拆分小于拆分点的数据和大于拆分点的数据
        smaller_data = train_data[train_data[feature] < Division_point]
        greater_data = train_data[train_data[feature] >= Division_point]
        """
        del smaller_data[feature]
        del greater_data[feature]
        """
        if len(smaller_data.columns) > 1:
            train(smaller_data, [feature, 1, Division_point])
        if len(greater_data.columns) > 1:
            train(greater_data, [feature, 0, Division_point])
    else:
        node_info.append([decision_parent, train_data.iloc[0]['species']])

# 测试函数        
def test(test_data):
    test_result = []
    n_true_result = 0
    # 逐一查找每个测试数据按照模型的输出值
    for index in range(len(test_data)):
        parent = None
        child = None
        while not isinstance(child, str):
            for info in node_info:
                if info[0] == parent:
                    if isinstance(info[1], str):
                        child = info[1]
                    else:
                        feature = info[1][0]
                        if info[1][1] == 1 and test_data.iloc[index][feature] < info[1][2]:
                            child = info[1]
                            parent = info[1]
                        if info[1][1] == 0 and test_data.iloc[index][feature] >= info[1][2]:
                            child = info[1]
                            parent = info[1]
        test_result.append(child)
        if child == test_data.iloc[index]['species']:
            n_true_result += 1  
    test_data['test_result'] = test_result
    acc = n_true_result / len(test_data)
    return test_data, acc

# 五折交叉法计算结果
acc_list = []
fold = 3
for remains in range(fold):
    my_list = list(range(fold))
    my_list.remove(remains)
    train_indices = [i for i in indices if i % fold in my_list]
    test_indices = [i for i in indices if i % fold == remains]
    
    # 根据索引选择训练和测试数据
    train_data = df.iloc[train_indices]
    test_data = df.iloc[test_indices]
    
    train(train_data)
    result, acc = test(test_data)
    acc_list.append(acc)
acc_mean = sum(acc_list) / len(acc_list)
print(acc_mean)


            

