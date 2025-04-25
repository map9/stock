import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data.sampler import SubsetRandomSampler
import random
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm
import pickle
import gc
import concurrent.futures
import multiprocessing
from preprocess_tools import parse_stock_code, parse_stock_name, zscore_based_smoothing, STOCK_NAME_PREFIXES_NORMAL, STOCK_NAME_PREFIXES_NEW, STOCK_NAME_PREFIXES_CHANGE
import torch.multiprocessing as mp

# 设置共享策略
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

# 配置参数
class Config:
    data_path = "data/sh/"  # 数据目录
    scaler_path = "models/scalers/"  # 用于保存每个特征的标准化器
    model_path = "models/"  # 模型保存路径
    sequence_length = 20  # 输入序列长度（使用多少天的历史数据）
    prediction_horizon = 1  # 预测未来多少天
    batch_size = 64*12
    learning_rate = 0.00015
    num_epochs = 10 # 50
    validation_split = 0.2
    test_split = 0.1
    d_model = 128  # Transformer模型维度
    nhead = 8  # 注意力头数
    num_encoder_layers = 3
    dim_feedforward = 512  # 前馈网络维度
    dropout = 0.2
    early_stopping_patience = 10

    # 不同类型特征的分组
    price_features = []
    percentage_features = ['收盘价', '总股数', '涨跌幅', '振幅', '换手率', '成交量', '量比']
    ratio_features = ['市盈率TTM', '市销率TTM', '市现率TTM', '市净率']
    volume_features = []

    @classmethod
    def get_all_features(cls):
        return (cls.price_features + cls.percentage_features + 
                cls.ratio_features + cls.volume_features)

# 确保存在模型和标准化器保存目录
os.makedirs(Config.model_path, exist_ok=True)
os.makedirs(Config.scaler_path, exist_ok=True)

# 数据预处理工具
class DataPreprocessor:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.scalers = {}
        
        # 为不同类型的特征选择不同的标准化方法
        for feature in Config.price_features:
            self.scalers[feature] = MinMaxScaler()
            
        for feature in Config.percentage_features:
            self.scalers[feature] = StandardScaler()
            
        for feature in Config.ratio_features:
            self.scalers[feature] = RobustScaler()  # 对异常值更稳健
            
        for feature in Config.volume_features:
            self.scalers[feature] = RobustScaler()  # 成交量可能有极端值
    
    def fit(self, df: pd.DataFrame):
        """对训练数据拟合所有标准化器"""
        for feature, scaler in self.scalers.items():
            if feature in df.columns:
                # 将数据整形为2D，适合标准化器
                scaler.fit(df[feature].values.reshape(-1, 1))
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据框的所有特征"""
        df_scaled = df.copy()
        for feature, scaler in self.scalers.items():
            if feature in df.columns:
                df_scaled[feature] = scaler.transform(df[feature].values.reshape(-1, 1))
        return df_scaled
    
    def inverse_transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """反向转换指定的特征"""
        df_inverse = df.copy()
        for feature in features:
            if feature in self.scalers and feature in df.columns:
                df_inverse[feature] = self.scalers[feature].inverse_transform(
                    df[feature].values.reshape(-1, 1)
                )
        return df_inverse
    
    def save(self):
        """保存标准化器到一个文件"""
        os.makedirs(Config.scaler_path, exist_ok=True)
        scaler_path = f"{Config.scaler_path}{self.stock_code}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    @classmethod
    def load(cls, stock_code: str):
        """从一个文件加载标准化器"""
        scaler_path = f"{Config.scaler_path}{stock_code}.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件 {scaler_path} 不存在")
        
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        preprocessor = cls(stock_code)
        preprocessor.scalers = scalers
        return preprocessor

# 股票数据集合
class StockOriginDataset:
    def __init__(self):
        # 获取所有股票代码
        stock_files = glob.glob(os.path.join(Config.data_path, "*.csv"))
        self.stock_codes = [os.path.basename(file).replace('.csv', '') for file in stock_files]
        
        # for testing purposes, limit to 100 stocks
        self.stock_codes = self.stock_codes[:100]
        
        self.stock_df_map = {}
    
    @staticmethod
    def process_batch(stock_codes_batch):
        """处理一批股票代码"""
        result = {}
        for code in stock_codes_batch:
            try:
                df = StockOriginDataset.load_stock_data(code)
                if df is not None:
                    result[code] = df
            except Exception as e:
                print(f"加载股票 {code} 数据时出错: {e}")
        return result

    
    def load(self):
        """加载所有股票数据，使用批处理优化的多进程方式"""
        self.unload()

        # 将股票代码分组，每组处理一批股票
        batch_size = 10
        # 根据 batch_size 和 CPU 核心数动态调整 max_workers
        max_workers = min(multiprocessing.cpu_count(), len(self.stock_codes) // (batch_size * 2))
        stock_batches = [self.stock_codes[i:i+batch_size] for i in range(0, len(self.stock_codes), batch_size)]

        # 使用进程池处理每一批
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(StockOriginDataset.process_batch, batch) for batch in stock_batches]

            # 展示进度条
            progress_bar = tqdm(total=len(self.stock_codes), desc="加载股票数据")

            # 处理结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    # 更新数据字典
                    self.stock_df_map.update(batch_results)
                    # 更新进度条
                    progress_bar.update(len(batch_results))
                except Exception as e:
                    print(f"处理进程任务时出错: {e}")

            progress_bar.close()

        return self.stock_df_map
    
    def unload(self):
        """卸载所有股票数据"""
        for stock_df in self.stock_df_map:
            del stock_df
        self.stock_df_map = {}
        gc.collect()

    @staticmethod
    def load_stock_data(stock_code: str) -> pd.DataFrame:
        """加载指定股票的数据"""
        file_path = os.path.join(Config.data_path, f"{stock_code}.csv")
        if not os.path.exists(file_path):
            return None            
        #print(f"正在加载股票数据: {file_path}")
        
        with open(file_path, 'r', encoding='gbk') as f:
            df = pd.read_csv(f)
        
        df['date'] = pd.to_datetime(df['交易日期'])
        # 将日期列设置为索引
        df.set_index('date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        # 主板、创业板、科创板的注册制新股上市前 5 个交易日均不设涨跌幅限制
        # 主板：第 6 个交易日起恢复 10% 涨跌幅限制。
        # 以上1 ～ 5日的数据记录先删除
        df = df.drop(df.index[:6])
        # 中国股票市场将涨跌幅限制调整为 10% 的上下限的时间点是 1996 年 12 月 16 日。
        # 根据沪深交易所的规定，自该日起，A 股和基金的交易价格涨跌幅限制统一设定为 10%，旨在抑制过度投机、稳定市场波动。
        # 这一制度沿用至今，适用于主板股票的正常交易日（新股上市首日、退市整理期等特殊情况除外）。
        df = df[df['交易日期'] >= '1996-12-16']

        # 1. '股票代码'
        stock_code_prefix, stock_code_prefix_value = parse_stock_code(stock_code)

        if df.empty:
            return None

        # 2. '股票名称'
        stock_previous_name = ''
        df['交易事件'] = STOCK_NAME_PREFIXES_NORMAL
        for row in df.itertuples():
            stock_name, stock_name_prefixe, stock_name_prefixe_value = parse_stock_name(row.股票名称)
            # 创业板 / 科创板：第 6 个交易日起涨跌幅限制为 20%
            # 将涨跌幅调整的为10%以内
            if (stock_code_prefix_value == 5) and (stock_code_prefix_value == 10):
                df.at[row.Index, '涨跌幅'] = df.at[row.Index, '涨跌幅'] / 2
                df.at[row.Index, '震幅'] = df.at[row.Index, '震幅'] / 2
            # 新股发行，早期新股发行不带前缀 'N'
            if row.Index == 0:
                stock_name_prefixe_value = STOCK_NAME_PREFIXES_NEW | stock_name_prefixe_value
            else:
                if stock_name == stock_previous_name or (stock_previous_name in stock_name) or (stock_name in stock_previous_name):
                    pass
                else:
                    stock_name_prefixe_value = STOCK_NAME_PREFIXES_CHANGE | stock_name_prefixe_value
            stock_previous_name = stock_name
            df.at[row.Index, '交易事件'] = stock_name_prefixe_value
            #print(stock_name, stock_name_prefixe, stock_name_prefixe_value)

        # 3. '开盘价', '最高价', '最低价', '收盘价'

        # 4. '成交量'
        # 对成交量先做对数变换
        df['成交量'] = np.log1p(df['成交量'])
        
        # 5. '换手率'

        # 6. '流通市值', '总市值'
        # 6.1 基于流通市值、总市值计算流通股数和总股数
        df['outstanding_shares'] = df['流通市值'] / df['收盘价']
        df['total_shares'] = df['总市值'] / df['收盘价']
        # 6.2 对流通股数、总股数使用分段平滑 Z-score
        df['outstanding_shares_std'] = zscore_based_smoothing(df['outstanding_shares'], 3)
        df['total_shares_std'] = zscore_based_smoothing(df['total_shares'], 3)
        # 6.3 对计算流通比，并使用分段平滑 Z-score
        df['outstanding_shares_ratio'] = df['outstanding_shares'] / df['total_shares']
        df['流通比'] = zscore_based_smoothing(df['outstanding_shares_ratio'], 3)
        # 6.4 对流通股数、总股数使用对数变换
        df['总股数'] = np.log1p(df['total_shares_std'])

        # 7. '市盈率TTM', '市销率TTM', '市现率TTM', '市净率'
        # 处理缺失值和无穷值，将无穷值和缺失值替换为0
        for col in ['市盈率TTM', '市销率TTM', '市现率TTM', '市净率']:
            df[col] = df[col].replace([np.nan, np.inf, -np.inf], 0)
        
        # 8. '量比'
        # 9. '涨跌幅', '振幅'

        # 数据类型转换
        feature_columns = ['交易事件', '收盘价', '总股数', '振幅', '换手率', '成交量', '量比', '市盈率TTM', '市销率TTM', '市现率TTM', '市净率', '涨跌幅']
        for col in feature_columns:
            df[col] = df[col].astype(np.float32)
        # 数据裁剪
        df = df[feature_columns]

        return df

# 股票数据集类
class StockDataset(Dataset):
    def __init__(
        self, 
        originDataset: StockOriginDataset,
        sequence_length: int = Config.sequence_length,
        prediction_horizon: int = Config.prediction_horizon,
        train: bool = True,
        validation: bool = False
    ):
        self.originDataset = originDataset
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.preprocessors = {}
        self.preprocessed_data = {}
        self.train = train
        self.validation = validation
        self.stock_index_map = {}
        
        if train:
            dataset_type = 'Train'
        elif validation:
            dataset_type = 'Validation'
        else:
            dataset_type = 'Test'
        print(f"正在处理 {dataset_type} 数据集...")

        # 将股票数据分批
        stock_data_items = list(originDataset.stock_df_map.items())
        # 每批处理的股票数量
        batch_size = 20
        # 根据 batch_size 和 CPU 核心数动态调整 max_workers
        max_workers = min(multiprocessing.cpu_count(), len(stock_data_items) // (batch_size * 2))
        batches = [stock_data_items[i:i+batch_size] for i in range(0, len(stock_data_items), batch_size)]
        
        # 使用进程池处理数据批次
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_stock_batch, batch, train=self.train, validation=self.validation) for batch in batches]
            
            # 展示进度条
            progress_bar = tqdm(total=len(stock_data_items), desc="加载股票数据")

            # 处理结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    # 更新数据字典
                    all_results.extend(batch_results)
                    # 更新进度条
                    progress_bar.update(len(batch_results))
                except Exception as e:
                    print(f"处理进程任务时出错: {e}")
        
        # 组合处理结果
        stock_start_idx = 0
        for result in all_results:
            stock_code = result['stock_code']
            self.preprocessors[stock_code] = result['preprocessor']
            self.preprocessed_data[stock_code] = result['df_scaled']
            valid_length = len(result['df_scaled']) - sequence_length - prediction_horizon + 1
            if valid_length > 0:
                self.stock_index_map[stock_code] = {
                    'start': stock_start_idx,
                    'end': stock_start_idx + valid_length - 1
                }
                stock_start_idx += valid_length
        # 清理内存
        #gc.collect()

    @staticmethod
    def process_stock_batch(batch_items, train: bool = True, validation: bool = False):
        """处理一批股票数据"""
        results = []
        for stock_code, df in batch_items:
            # 拆分训练、验证和测试集
            train_size = int(len(df) * (1 - Config.validation_split - Config.test_split))
            val_size = int(len(df) * Config.validation_split)
            
            if train:
                df_subset = df[:train_size]
            elif validation:
                df_subset = df[train_size:train_size+val_size]
            else:  # 测试集
                df_subset = df[train_size+val_size:]
            
            # 如果子集太小，跳过
            if len(df_subset) <= Config.sequence_length + Config.prediction_horizon:
                continue

            # 标准化处理
            if train:
                # 训练模式下创建并拟合预处理器
                preprocessor = DataPreprocessor(stock_code)
                preprocessor.fit(df_subset[Config.get_all_features()])
                preprocessor.save()
            else:
                # 验证或测试模式下加载已保存的预处理器
                preprocessor = DataPreprocessor.load(stock_code)
            
            df_scaled = preprocessor.transform(df_subset[Config.get_all_features()])
            results.append({
                'stock_code': stock_code,
                'preprocessor': preprocessor,
                'df_scaled': df_scaled
            })
                
        return results

    def __len__(self):
        # 所有有效样本的总数
        return sum(idx_map['end'] - idx_map['start'] + 1 for idx_map in self.stock_index_map.values())
    
    def __getitem__(self, idx):
        # 找出idx对应哪只股票及其在该股票中的位置
        stock_code = None
        local_idx = None
        
        for code, range_map in self.stock_index_map.items():
            if range_map['start'] <= idx <= range_map['end']:
                stock_code = code
                local_idx = idx - range_map['start']
                break
        
        if stock_code is None:
            raise IndexError(f"索引 {idx} 超出范围")
        
        # 获取这只股票的处理后数据
        df_scaled = self.preprocessed_data[stock_code]
        
        # 提取输入序列和目标
        X = df_scaled.iloc[local_idx:local_idx+self.sequence_length].values
        y = df_scaled.iloc[local_idx+self.sequence_length:local_idx+self.sequence_length+self.prediction_horizon]['涨跌幅'].values
        
        # 转换为张量
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return X, y, stock_code

# Transformer模型定义
class StockTransformer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        d_model: int = Config.d_model,
        nhead: int = Config.nhead,
        num_encoder_layers: int = Config.num_encoder_layers,
        dim_feedforward: int = Config.dim_feedforward,
        dropout: float = Config.dropout
    ):
        super(StockTransformer, self).__init__()
        
        # 特征映射层
        self.feature_projection = nn.Linear(feature_size, d_model)
        
        # 位置编码（可学习的）
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, Config.sequence_length, d_model)
        )
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 指定输入为 [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, Config.prediction_horizon)
        )
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, feature_size]
        
        # 特征映射
        x = self.feature_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # Transformer编码
        # 不需要显式的mask，因为我们使用的是自注意力
        encoded = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 使用序列的最后一个位置进行预测
        out = self.predictor(encoded[:, -1])  # [batch_size, prediction_horizon]
        
        return out

# 新增：清理共享内存函数
def cleanup_shm():
    """显式清理 PyTorch 共享内存段"""
    import torch.multiprocessing as mp
    if hasattr(mp, 'resource_tracker'):
        if hasattr(mp.resource_tracker, 'shm_registry'):
            # 清除所有共享内存注册
            with mp.resource_tracker.get_resource_tracker()._lock:
                mp.resource_tracker.get_resource_tracker().shm_registry.clear()

# 训练和验证函数
def train_model(device, stock_origin_dataset: StockOriginDataset, retrain: bool = True):
    # 创建数据集和数据加载器
    train_dataset = StockDataset(stock_origin_dataset, train=True, validation=False)
    val_dataset = StockDataset(stock_origin_dataset, train=False, validation=True)
    # 打印数据集大小
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=min(4, multiprocessing.cpu_count()//2),
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=min(4, multiprocessing.cpu_count()//2),
        pin_memory=True,
        persistent_workers=False
    )

    # 获取特征数量
    sample_X, _, _ = train_dataset[0]
    feature_size = sample_X.shape[1]
    
    # 初始化模型
    model = StockTransformer(feature_size=feature_size)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 检查是否需要加载已有模型
    best_val_loss = float('inf')
    early_stopping_counter = 0
    start_epoch = 0
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    if retrain and os.path.exists(f"{Config.model_path}best_model.pth"):
        print("检测到已有模型，正在加载...")
        checkpoint = torch.load(f"{Config.model_path}best_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        start_epoch = checkpoint['epoch'] + 1
        print(f"加载完成，从第 {start_epoch} 轮开始训练，最佳验证损失为 {val_losses[-1]:.4f}")
    
    
    for epoch in range(start_epoch, Config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]")
        for X, y, _ in progress_bar:
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止爆炸
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
                
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Val]")
            for X, y, _ in progress_bar:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{Config.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f"{Config.model_path}best_model.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= Config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f"{Config.model_path}loss_curve{epoch}.png")
        plt.close()

    # 清理共享内存
    #cleanup_shm()        
    return model

# 评估函数
def evaluate_model(device, stock_origin_dataset: StockOriginDataset):
    # 创建测试数据集
    test_dataset = StockDataset(stock_origin_dataset, train=False, validation=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False,
        num_workers=min(4, multiprocessing.cpu_count()//2),
        pin_memory=True
    )
    
    # 获取特征数量
    sample_X, _, _ = test_dataset[0]
    feature_size = sample_X.shape[1]
    
    # 初始化模型
    model = StockTransformer(feature_size=feature_size)
    model = model.to(device)
    
    # 加载最佳模型
    checkpoint = torch.load(f"{Config.model_path}best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    # 用于存储预测和真实值
    all_preds = []
    all_targets = []
    all_stock_codes = []
    
    # 评估模型
    with torch.no_grad():
        for X, y, stock_code in tqdm(test_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            # 保存预测和真实值
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_stock_codes.extend(stock_code)
    
    # 合并结果
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 计算整体MSE
    mse = np.mean((all_preds - all_targets) ** 2)
    print(f"测试集MSE: {mse:.4f}")
    
    # 按股票计算MSE
    stock_wise_mse = {}
    unique_stocks = list(set(all_stock_codes))
    
    for stock in unique_stocks:
        stock_indices = [i for i, code in enumerate(all_stock_codes) if code == stock]
        if stock_indices:
            stock_preds = all_preds[stock_indices]
            stock_targets = all_targets[stock_indices]
            stock_mse = np.mean((stock_preds - stock_targets) ** 2)
            stock_wise_mse[stock] = stock_mse
    
    # 打印每只股票的MSE
    print("\n按股票的MSE:")
    for stock, mse in sorted(stock_wise_mse.items(), key=lambda x: x[1]):
        print(f"{stock}: {mse:.4f}")
    
    # 绘制预测与实际值的散点图
    plt.figure(figsize=(10, 10))
    plt.scatter(all_targets.flatten(), all_preds.flatten(), alpha=0.3)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel('Actual values')
    plt.ylabel('predicted values')
    plt.title('Prediction vs Actual')
    plt.savefig(f"{Config.model_path}prediction_vs_actual.png")
    plt.close()
    
    return mse, stock_wise_mse

# 预测单只股票的未来涨跌幅
def predict_next_day(device, stock_code, num_past_days=Config.sequence_length):
    """
    使用训练好的模型预测指定股票的下一个交易日涨跌幅
    
    参数:
    stock_code (str): 股票代码
    num_past_days (int): 使用多少天的历史数据
    
    返回:
    float: 预测的涨跌幅
    """
    df = StockOriginDataset.load_stock_data(stock_code)
    if df is None:
        raise FileNotFoundError(f"股票 {stock_code} 数据文件不存在或者数据不足支撑预测。")

    # 只保留最新的num_past_days天数据
    df = df.tail(num_past_days)
    if len(df) < num_past_days:
        raise ValueError(f"股票 {stock_code} 的历史数据不足 {num_past_days} 天")
    
    # 加载预处理器
    preprocessor = DataPreprocessor.load(stock_code)
    
    # 预处理数据
    df_scaled = preprocessor.transform(df[Config.get_all_features()])
    
    # 转换为张量
    X = torch.FloatTensor(df_scaled.values).unsqueeze(0)  # 添加批次维度
    
    # 加载模型
    sample_X = X  # 用来获取特征数量
    feature_size = sample_X.shape[2]
    
    model = StockTransformer(feature_size=feature_size)
    checkpoint = torch.load(f"{Config.model_path}best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 进行预测
    with torch.no_grad():
        X = X.to(device)
        prediction = model(X)
    
    # 将预测值从标准化后的值转换回实际涨跌幅
    scaled_prediction = prediction.cpu().numpy()[0]
    
    # 创建一个DataFrame来存储标准化后的预测值
    pred_df = pd.DataFrame({'涨跌幅': scaled_prediction})
    
    # 反向转换
    actual_prediction = preprocessor.inverse_transform(pred_df, ['涨跌幅'])['涨跌幅'].values[0]
    
    return actual_prediction

# 主函数
def main():
    # 确保在M1 Mac上可以使用GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 加载股票数据
    print("1. 开始加载股票数据 ...")
    stock_origin_dataset = StockOriginDataset()
    stock_origin_dataset.load()
    print(f"加载了 {len(stock_origin_dataset.stock_df_map)} / {len(stock_origin_dataset.stock_codes)} 只股票数据。")
    
    # 如果不强制要求为fork方法，可能会导致在Mac上使用Dataloader出现不删除torch_shm_manager的错误
    # 导致出现too many open files的错误
    # MAC OS默认为spawn方法
    # 所有其他的方法都是错误的
    multiprocessing.set_start_method("fork", force=True)

    # 训练模型
    print(f"\n2. 使用设备: {device}，开始训练模型 ...")
    model = train_model(device, stock_origin_dataset)
    
    # 评估模型
    print(f"\n3. 使用设备: {device}，开始评估模型 ...")
    mse, _ = evaluate_model(device, stock_origin_dataset)
    
    # 示例：预测特定股票的下一个交易日涨跌幅
    # 注意：请替换为实际存在的股票代码
    try:
        # 假设我们有一个名为"sh000001"的股票代码
        stock_to_predict = "sh600812"
        next_day_return = predict_next_day(device, stock_to_predict)
        print(f"\n股票 {stock_to_predict} 下一个交易日的预测涨跌幅: {next_day_return:.2%}")
    except Exception as e:
        print(f"预测时发生错误: {e}")

if __name__ == "__main__":
    main()