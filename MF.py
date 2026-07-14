import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据
def load_movielens_100k(filepath=r'ml-100k/u.data'):
    df = pd.read_csv(filepath, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    R_full = np.zeros((n_users, n_items))
    for row in df.itertuples():
        R_full[row.user - 1, row.item - 1] = row.rating
    return R_full, df, n_users, n_items

# 2. 矩阵分解训练（增加测试集参数）
def sgd_mf_with_bias(R_train, R_full, test_mask, K, steps=50, gamma=0.005, lam=0.02):
    M, N = R_train.shape
    mu = np.mean(R_train[R_train > 0])
    b_u = np.zeros(M)
    b_i = np.zeros(N)
    P = np.random.normal(0, 0.1, (M, K))
    Q = np.random.normal(0, 0.1, (N, K))

    train_entries = np.argwhere(R_train > 0)
    total_samples = len(train_entries)

    for step in range(steps):
        np.random.shuffle(train_entries)
        total_loss = 0.0

        for i, j in train_entries:
            r_ij = R_train[i, j]
            pred = mu + b_u[i] + b_i[j] + np.dot(P[i], Q[j])
            e = r_ij - pred

            p_i = P[i].copy()
            q_j = Q[j].copy()
            b_u_i = b_u[i]
            b_i_j = b_i[j]

            b_u[i] += gamma * (e - lam * b_u_i)
            b_i[j] += gamma * (e - lam * b_i_j)
            P[i] = p_i + gamma * (e * q_j - lam * p_i)
            Q[j] = q_j + gamma * (e * p_i - lam * q_j)

            total_loss += e ** 2 + lam * (b_u_i ** 2 + b_i_j ** 2 + np.dot(p_i, p_i) + np.dot(q_j, q_j))

        # 计算训练集平均损失（仍用于监控）
        avg_loss = total_loss / total_samples

        # ---- 每一轮结束后，计算测试集指标 ----
        R_pred = mu + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P @ Q.T
        test_true = R_full[test_mask]
        test_pred = R_pred[test_mask]
        rmse_test = np.sqrt(np.mean((test_true - test_pred) ** 2))
        mae_test = np.mean(np.abs(test_true - test_pred))

        print(f"Epoch {step+1}/{steps}, Loss: {avg_loss:.4f}, Test RMSE: {rmse_test:.4f}, Test MAE: {mae_test:.4f}")

    return mu, b_u, b_i, P, Q

# 预测函数（不变）
def predict(mu, b_u, b_i, P, Q):
    return mu + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P @ Q.T

# 3. 主程序
if __name__ == "__main__":
    R_full, df, n_users, n_items = load_movielens_100k()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    R_train = np.zeros((n_users, n_items))
    for row in train_df.itertuples():
        R_train[row.user - 1, row.item - 1] = row.rating

    test_mask = np.zeros((n_users, n_items), dtype=bool)
    for row in test_df.itertuples():
        test_mask[row.user - 1, row.item - 1] = True

    print(f"训练集: {len(train_df)}条, 测试集: {len(test_df)}条")

    # 训练，传入测试集数据
    mu, b_u, b_i, P, Q = sgd_mf_with_bias(
        R_train, R_full, test_mask,
        K=20, steps=50, gamma=0.005, lam=0.02
    )

    # 最终测试集评估
    R_pred = predict(mu, b_u, b_i, P, Q)
    test_true = R_full[test_mask]
    test_pred = R_pred[test_mask]
    rmse_final = np.sqrt(np.mean((test_true - test_pred) ** 2))
    mae_final = np.mean(np.abs(test_true - test_pred))
    print(f"\n最终测试集 RMSE = {rmse_final:.4f}, MAE = {mae_final:.4f}")