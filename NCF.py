import numpy as np
import pandas as pd # 处理表格数据DataFrame
import torch
import torch.nn as nn # 提供层
import torch.optim as optim # 提供优化器
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # 导入切分训练集、数据集的函数

# 读取数据
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('ml-100k/u.data', sep='\t', names=columns) # data是一个dataFrame。

# 转换为隐式反馈：评分 >= 4 为正样本
data['label'] = (data['rating'] >= 4).astype(int) # bool转换为int。在data中再加上一列

# 重新索引，使 user 和 item 从 0 开始
user_ids = data['user_id'].unique() # 返回Numpy数组，方便顺序迭代
item_ids = data['item_id'].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {i: j for j, i in enumerate(item_ids)}
data['user'] = data['user_id'].map(user2idx) # 替换成连续索引，符合embedding的输入要求
data['item'] = data['item_id'].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)
print(f"用户数: {num_users}, 物品数: {num_items}, 交互数: {len(data)}")

# 划分训练集/测试集
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42) # 保证每次划分结果一样。

# Dataset 定义
class MovieLensDataset(Dataset):
    def __init__(self, df): # 取3列构建pytorch张量，来按索引取样本
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.items = torch.tensor(df['item'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# ------------------ 2. NeuMF 模型定义 ------------------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_layers=[32, 16, 8], dropout=0.2):
        """
        mf_dim: GMF部分的嵌入维度
        mlp_layers: MLP各层输出维度，第一层输入是 user/item 嵌入的拼接
        """
        super(NeuMF, self).__init__() # 执行nn.module()的构造函数

        # GMF分支，广义矩阵分解
        self.mf_user_embed = nn.Embedding(num_users, mf_dim) # 神经网络层——查找表
        self.mf_item_embed = nn.Embedding(num_items, mf_dim)

        # MLP分支
        mlp_embed_dim = mlp_layers[0] // 2
        self.mlp_user_embed = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embed = nn.Embedding(num_items, mlp_embed_dim)

        # 构建 MLP 层
        mlp_modules = []
        input_dim = mlp_layers[0]  # 拼接后的维度
        for out_dim in mlp_layers[1:]:
            mlp_modules.append(nn.Linear(input_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout)) # 随机将一部分神经元输出置0，防止过拟合
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules) # 将多个层对象解包，再串联，完成前向传播

        # 融合层
        self.predict = nn.Sequential(
            nn.Linear(mf_dim + mlp_layers[-1], 1),
            nn.Sigmoid()   # 输出 0~1 概率
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules(): # 遍历当下所有的子模块
            if isinstance(m, nn.Embedding): # 对Embedding和Linear层做特殊初始化
                nn.init.xavier_uniform_(m.weight) # 使用Xavier均匀分布初始化权重
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) # 偏执初始化为0。

    def forward(self, user_ids, item_ids):
        # GMF
        mf_user = self.mf_user_embed(user_ids) # 输入索引，从mf_user_embed里面取对应行
        mf_item = self.mf_item_embed(item_ids)
        mf_vec = mf_user * mf_item

        # MLP
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_vec = torch.cat([mlp_user, mlp_item], dim=-1) # 按最后一维拼接
        mlp_out = self.mlp(mlp_vec)

        # 融合
        concat = torch.cat([mf_vec, mlp_out], dim=-1)
        output = self.predict(concat).squeeze() # 16维变成1维
        return output

# ------------------ 3. 评估指标：HR@10, NDCG@10 ------------------
def compute_metrics(model, test_df, device, k=10):
    """计算 Hit Rate@k 和 NDCG@k"""
    model.eval() # 评估模式，关闭dropout
    # 构建用户-正样本字典
    user_pos_items = {} # 用户真实喜欢集合
    for _, row in test_df.iterrows(): # iterrows()逐行遍历表格，评估指标在测试集上计算
        u = row['user']
        i = row['item']
        label = row['label']
        if label == 1:
            if u not in user_pos_items:
                user_pos_items[u] = set()
            user_pos_items[u].add(i)

    hits = 0.0
    ndcg = 0.0
    total_users = len(user_pos_items)
    if total_users == 0:
        return 0, 0

    all_items = torch.arange(num_items, dtype=torch.long).to(device)

    for u, pos_set in user_pos_items.items(): # .items()返回键值对
        user_tensor = torch.full((num_items,), u, dtype=torch.long).to(device)
        with torch.no_grad(): # 只进行预测
            scores = model(user_tensor, all_items).cpu().numpy() # 一个用户对所有物品的喜欢概率。nn.module会自动调用forward。
        # 排序取 top-k
        item_scores = list(zip(range(num_items), scores)) # zip将两个序列按位置配对，打包成元组
        item_scores.sort(key=lambda x: x[1], reverse=True) # 将分数从大到小排序
        top_k_items = [item for item, score in item_scores[:k]] # 提取物品ID

        # Hit Ratio
        if len(pos_set.intersection(top_k_items)) > 0:
            hits += 1.0 # 看是否命中

        # NDCG
        dcg = 0.0 # 折损累积增益
        idcg = 0.0 # 理想DCG
        for rank, item in enumerate(top_k_items, start=1):
            if item in pos_set:
                dcg += 1 / np.log2(rank + 1)
        for rank in range(1, min(len(pos_set), k) + 1):
            idcg += 1 / np.log2(rank + 1)
        ndcg += dcg / idcg if idcg > 0 else 0

    return hits / total_users, ndcg / total_users

# ------------------ 4. 训练配置 ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

batch_size = 256
epochs = 20
lr = 1e-3

train_dataset = MovieLensDataset(train_df)
test_dataset = MovieLensDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle=True，打乱顺序
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = NeuMF(num_users, num_items, mf_dim=8, mlp_layers=[32, 16, 8], dropout=0.2).to(device)
criterion = nn.BCELoss() # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=lr) # Adam优化器，自适应调整学习率

# ------------------ 5. 训练循环 ------------------
for epoch in range(epochs):
    model.train() # 开启DropOut
    total_loss = 0
    for users, items, labels in train_loader:
        users, items, labels = users.to(device), items.to(device), labels.to(device) # 移动到GPU

        preds = model(users, items)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(users)

    train_loss = total_loss / len(train_dataset)

    # 测试集 loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, items, labels in test_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)
            test_loss += loss.item() * len(users)
    test_loss /= len(test_dataset)

    # 评估 HR@10 和 NDCG@10
    hr10, ndcg10 = compute_metrics(model, test_df, device, k=10)

    print(f"Epoch {epoch+1:2d}/{epochs} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | HR@10: {hr10:.4f} | NDCG@10: {ndcg10:.4f}")

print("\n训练完成！")
print(f"GPU 显存已分配: {torch.cuda.memory_allocated()/1024**2:.2f} MB")