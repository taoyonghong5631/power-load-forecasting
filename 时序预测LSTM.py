# -*- coding: utf-8 -*-
"""
电力负荷短期预测与异常检测
数据集：UCI ElectricityLoadDiagrams20112014
模型：LSTM + 递归多步预测
可视化：Plotly
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'data_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip',
    'data_dir': './data',
    'zip_path': './data/LD2011_2014.txt.zip',
    'txt_path': './data/LD2011_2014.txt',
    'client_col': 1,               # 使用第2列（MT_001）作为示例，0是时间列
    'resample_freq': 'H',          # 重采样为小时级
    'look_back': 24,               # 输入窗口：过去24小时
    'forecast_horizon': 24,        # 预测未来24小时
    'train_ratio': 0.8,
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])


# ==================== 数据下载与加载 ====================
def download_data():
    """下载并解压 UCI 数据集，若失败则生成模拟数据"""
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    if not os.path.exists(CONFIG['txt_path']):
        try:
            print("正在下载数据集...")
            urllib.request.urlretrieve(CONFIG['data_url'], CONFIG['zip_path'])
            with zipfile.ZipFile(CONFIG['zip_path'], 'r') as zip_ref:
                zip_ref.extractall(CONFIG['data_dir'])
            print("下载并解压完成。")
        except Exception as e:
            print(f"下载失败：{e}，将生成模拟数据。")
            return generate_synthetic_data()
    # 读取数据（只读取时间列和指定的客户列）
    print("正在读取数据...")
    df = pd.read_csv(
        CONFIG['txt_path'],
        sep=';',
        decimal=',',
        index_col=0,
        parse_dates=True,
        usecols=[0, CONFIG['client_col']]
    )
    # 重命名列
    df.columns = ['load']
    return df


def generate_synthetic_data():
    """生成模拟的电力负荷数据，用于演示"""
    print("生成模拟电力负荷数据...")
    date_rng = pd.date_range(start='2012-01-01', end='2014-12-31', freq='H')
    n = len(date_rng)
    t = np.arange(n)
    # 模拟日周期、周周期、年周期及噪声
    daily = 10 * np.sin(2 * np.pi * t / 24)
    weekly = 5 * np.sin(2 * np.pi * t / (24 * 7))
    yearly = 20 * np.sin(2 * np.pi * t / (24 * 365))
    trend = 0.001 * t
    noise = np.random.normal(0, 2, n)
    load = 50 + daily + weekly + yearly + trend + noise
    df = pd.DataFrame({'load': load}, index=date_rng)
    return df


def preprocess_data(df):
    """数据清洗、重采样、插值"""
    print("数据预处理...")
    # 重采样为小时级，取均值
    df_hourly = df.resample(CONFIG['resample_freq']).mean()
    # 处理缺失值：线性插值
    df_hourly = df_hourly.interpolate(method='linear')
    # 处理异常值：使用 3σ 原则替换为均值（可选）
    mean = df_hourly['load'].mean()
    std = df_hourly['load'].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    df_hourly['load'] = np.where(
        (df_hourly['load'] < lower_bound) | (df_hourly['load'] > upper_bound),
        mean, df_hourly['load']
    )
    return df_hourly


# ==================== 数据集构建 ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, look_back):
        self.data = data
        self.look_back = look_back

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.look_back]
        y = self.data[idx + self.look_back]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


# ==================== LSTM 模型 ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# ==================== 递归多步预测 ====================
def recursive_forecast(model, initial_history, horizon, look_back, device):
    """
    递归多步预测
    :param model: 训练好的模型
    :param initial_history: 初始历史序列，长度 >= look_back，numpy array
    :param horizon: 预测步数
    :param look_back: 输入窗口大小
    :return: 预测序列
    """
    model.eval()
    history = list(initial_history[-look_back:])  # 取最后 look_back 个点
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(history[-look_back:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).cpu().item()
            predictions.append(pred)
            history.append(pred)
    return np.array(predictions)


# ==================== 评估与可视化 ====================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def plot_results(train_dates, train_values, test_dates, test_values, predictions, look_back, scaler):
    """使用 Plotly 绘制预测结果和残差分布"""
    # 反标准化
    test_values_inv = scaler.inverse_transform(test_values.reshape(-1, 1)).flatten()
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    # 计算残差
    residuals = test_values_inv - predictions_inv
    residual_std = np.std(residuals)
    # 置信区间 (95%)
    upper = predictions_inv + 1.96 * residual_std
    lower = predictions_inv - 1.96 * residual_std

    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('历史与预测负荷', '残差分布', '残差自相关', '预测区间'),
        specs=[[{"colspan": 2}, None], [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # 1. 历史与预测负荷
    # 为了展示，只取测试集前 200 个点
    n_show = min(200, len(test_dates))
    fig.add_trace(
        go.Scatter(x=train_dates[-100:], y=scaler.inverse_transform(train_values[-100:].reshape(-1,1)).flatten(),
                   mode='lines', name='训练集历史', line=dict(color='gray')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=test_dates[:n_show], y=test_values_inv[:n_show],
                   mode='lines', name='真实值', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=test_dates[:n_show], y=predictions_inv[:n_show],
                   mode='lines', name='预测值', line=dict(color='red')),
        row=1, col=1
    )
    # 置信区间
    fig.add_trace(
        go.Scatter(x=test_dates[:n_show], y=upper[:n_show],
                   mode='lines', line=dict(width=0), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=test_dates[:n_show], y=lower[:n_show],
                   mode='lines', line=dict(width=0), fill='tonexty',
                   fillcolor='rgba(255,0,0,0.2)', name='95% 置信区间'),
        row=1, col=1
    )

    # 2. 残差分布
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=50, name='残差', marker_color='green'),
        row=2, col=1
    )

    # 3. 残差自相关（简单绘制残差序列）
    fig.add_trace(
        go.Scatter(x=test_dates, y=residuals, mode='lines', name='残差序列', line=dict(color='purple')),
        row=2, col=2
    )

    fig.update_layout(height=800, width=1200, title_text="电力负荷短期预测与残差分析")
    fig.show()
    return residuals


def detect_anomalies(residuals, threshold=3):
    """基于残差 3σ 原则检测异常"""
    std = np.std(residuals)
    mean = np.mean(residuals)
    anomalies = np.where(np.abs(residuals - mean) > threshold * std)[0]
    return anomalies


# ==================== 主流程 ====================
def main():
    # 1. 加载数据
    df = download_data()
    df = preprocess_data(df)
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    # 2. 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['load']]).flatten()

    # 3. 划分训练集和测试集
    train_size = int(len(data_scaled) * CONFIG['train_ratio'])
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]

    print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

    # 4. 创建 Dataset 和 DataLoader
    train_dataset = TimeSeriesDataset(train_data, CONFIG['look_back'])
    # 验证集取训练集最后 20%
    val_size = int(len(train_dataset) * 0.2)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 5. 模型初始化
    device = CONFIG['device']
    model = LSTMModel(
        input_size=1,
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    print(f"使用设备: {device}")

    # 6. 训练
    print("开始训练...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        device=device
    )

    # 7. 递归多步预测
    print("进行递归多步预测...")
    # 使用训练集最后 look_back 个点作为初始历史
    initial_history = train_data[-CONFIG['look_back']:]
    # 预测测试集长度，但为了演示，只预测前 forecast_horizon 步
    horizon = min(CONFIG['forecast_horizon'], len(test_data))
    predictions = recursive_forecast(
        model, initial_history, horizon, CONFIG['look_back'], device
    )

    # 取对应的真实值
    true_values = test_data[:horizon]
    test_dates_forecast = test_dates[:horizon]

    # 8. 评估
    # 注意：预测值是标准化的，需要反标准化
    pred_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_inv = scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    metrics = evaluate(true_inv, pred_inv)
    print("预测评估指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 9. 可视化
    print("生成可视化图表...")
    # 为了绘制完整测试集，我们可以用真实值滚动预测，但这里只展示前 horizon 步
    # 创建一个包含训练集和测试集预测的图
    residuals = plot_results(
        train_dates, train_data,
        test_dates_forecast, true_values,
        predictions, CONFIG['look_back'], scaler
    )

    # 10. 异常检测
    anomalies = detect_anomalies(residuals, threshold=3)
    print(f"检测到 {len(anomalies)} 个异常点（基于残差3σ）")
    if len(anomalies) > 0:
        print("异常点时间:", test_dates_forecast[anomalies])

    print("完成！")


if __name__ == '__main__':
    main()