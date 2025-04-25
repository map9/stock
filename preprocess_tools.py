import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plot_tools import plot_stock, plot_lines, plot
import gc

# 基于Z-score的方法，去除异常值
# 计算前面分段数据序列的Z-score，对于新数据点，计算其和前面分段数据序列的偏离值，超过指定的阈值，则认为该数据点是异常值
def zscore_based_smoothing(data, zscore_threshold=3):
  filtered_data = []
  current_segment = []
  for i, value in enumerate(data):
    if len(current_segment) < 2:
      # 前两个数据点直接加入当前分段
      current_segment.append(value)
    else:
      segment_mean = np.mean(current_segment)
      segment_std = np.std(current_segment)
      z_score = (value - segment_mean) / segment_std if segment_std != 0 else 0
      if abs(z_score) <= zscore_threshold:
        # 当前数据的Z - score未超过阈值，属于同一分段
        current_segment.append(value)
      else:
        # 进入新的分段，对当前分段进行平滑处理
        segment_mean = np.mean(current_segment)
        filtered_data.extend([segment_mean] * len(current_segment))
        current_segment = [value]
  # 处理最后一个分段
  segment_mean = np.mean(current_segment)
  filtered_data.extend([segment_mean] * len(current_segment))
  return filtered_data
  
# 股票名称前缀的组合规则定义
# [1][2][3][4] 00 00 00 00
# [1]: 00 NULL | 01 XD（除息） | 10 XR（除权） | 11 DR（除息+除权） 
# [2]: 00 NULL | 01 N（新股上市首日） | 10 C（注册制新股第2-5日） | 11 [自定义](更名，有可能意味着有资产重组)
# [3]: 00 NULL | 01 S（未股改，历史标识） | 10 G（已股改，历史标识）  
# [4]: 00 NULL | 01 ST（两年亏损风险警示） | 10 *ST（三年亏损+退市风险警示） | 11 PT（退市预警，历史标识） 
# 用四个字节来表示股票的状态
STOCK_NAME_PREFIXES_NORMAL = 0b00000000
STOCK_NAME_PREFIXES_CHANGE = 0b00110000
STOCK_NAME_PREFIXES_NEW = 0b00010000
STOCK_NAME_PREFIXES2VALUES = [  
  ['XDS*ST', 0b01000110],  
  ['XDSST', 0b01000101],  
  ['XR*ST', 0b10000010],  
  ['XD*ST', 0b01000010],  
  ['XRST', 0b10000001],  
  ['XDST', 0b01000001],  
  ['S*ST', 0b00000110],  
  ['G*ST', 0b00001010],  
  ['N*ST', 0b00010010],  
  ['NST', 0b00010001],  
  ['*ST', 0b00000010],  
  ['SST', 0b00000101],  
  ['DRG', 0b11001000],  
  ['GST', 0b00001001],  
  ['XDG', 0b01001000],  
  ['XDS', 0b01000100],  
  ['XRG', 0b10001000],  
  ['XRS', 0b10000100],  
  ['XR', 0b10000000],  
  ['XD', 0b01000000],  
  ['ST', 0b00000001],  
  ['DR', 0b11000000],  
  ['NG', 0b00011000],  
  ['PT', 0b00000011],  
  ['C', 0b00100000],  
  ['G', 0b00001000],  
  ['N', STOCK_NAME_PREFIXES_NEW],  
  ['S', 0b00000100],  
]  

# 解析'股票名称'中包含的特征，返回去除特征后的股票名称、特征代码以及特征数值
def parse_stock_name(stock_name):
  stock_name_clean = stock_name
  stock_name_prefixe = ''
  stock_name_prefixe_value = STOCK_NAME_PREFIXES_NORMAL
  for prefixes_value in STOCK_NAME_PREFIXES2VALUES:
    if stock_name.startswith(prefixes_value[0]):
      stock_name_clean = stock_name[len(prefixes_value[0]):]
      stock_name_prefixe = prefixes_value[0]
      stock_name_prefixe_value = prefixes_value[1]
      break # 已经找到前缀，跳出循环
  return stock_name_clean, stock_name_prefixe, stock_name_prefixe_value

# ['sh600', 'sh601', 'sh603', 'sh605', 'sh688', 'sh689', 'sz000', 'sz001', 'sz002', 'sz003', 'sz300', 'sz301', 'sz302']
STOCK_CODE_PREFIXES_INDEX = 0
STOCK_CODE_PREFIXES2VALUES = [
  ['sh000',  STOCK_CODE_PREFIXES_INDEX], # 指数
  ['sz399',  STOCK_CODE_PREFIXES_INDEX], # 指数
  ['sh600',  1], # 上交所主板大盘蓝筹股，通常是一些大型国有企业或知名企业
  ['sh601',  2], # 上交所主板上市较晚的蓝筹股
  ['sh603',  3], # 上交所主板中小盘股
  ['sh605',  4], # 上交所主板新兴行业企业，具有较高的成长潜力和创新能力
  ['sh688',  5], # 上交所科创板股票
  ['sh689',  5], # 上交所科创板股票
  ['sz000',  6], # 深交所主板股票，以传统产业为主
  ['sz001',  7], # 深交所主板股票，以传统产业为主，类似 000 开头的股票
  ['sz002',  8], # 深交所中小板股票，公司规模相对较小，但具有较高的成长性
  ['sz003',  9], # 深交所主板注册制股票
  ['sz3',   10], # 深交所创业板股票，主要针对高科技、高成长的中小企业，上市门槛相对较低，风险相对较大
  ['bj8',   11], # 北交所新三板创新层或基础层的股票
  ['bj4',   12], # 北交所新三板老三板股票，主要是从沪深两市退市后到三板交易的股票
  ['sh9',   13], # 上交所 B 股，以美元计价，供境外投资者交易
  ['sh2',   14], # 深交所 B 股，以港元计价，面向境外投资者
]

# 解析'股票代码'中包含的特征，返回去除特征后的股票代码、特征代码以及特征数值
def parse_stock_code(stock_code):
  stock_code_prefix = stock_code[:5]
  stock_code_prefix_value = 0
  for prefixes_value in STOCK_CODE_PREFIXES2VALUES:
    if stock_code_prefix.startswith(prefixes_value[0]):
      stock_code_prefix = prefixes_value[0]
      stock_code_prefix_value = prefixes_value[1]
      break
  return stock_code_prefix, stock_code_prefix_value

def check_stock_file(stock_file_path):
  stock_code = os.path.basename(stock_file_path).split('.')[0]
  df = pd.read_csv(stock_file_path, encoding = 'gbk')
  df['Date'] = pd.to_datetime(df['交易日期'])
  # 将日期列设置为索引
  df.set_index('Date', inplace=True)
  df.sort_index(ascending=True, inplace=True)

  # 统计分析股票交易数据中股票名称中的特征
  stock_code_prefix, _ = parse_stock_code(stock_code)
  unique_stock_names = []
  if '股票名称' in df.columns:
    stock_names = df['股票名称'].unique().tolist()
    for stock_name in stock_names:
      stock_name, _, _ = parse_stock_name(stock_name)
      #print(stock_name_prefixe, stock_name)
      if stock_name not in unique_stock_names:
        unique_stock_names.append(stock_name)

  # 打印结果
  print(f"股票代码: {stock_code}")
  print(f"股票代码前缀: {stock_code_prefix}")
  print(f"股票名称: {unique_stock_names}")

  return stock_code_prefix, unique_stock_names

def check_stock_dataset(data_dir):
  unique_stock_codes = []
  unique_stock_names = []
  for root, dirs, files in os.walk(data_dir):
    for file in sorted(files):
      if file.endswith('.csv'):
        print(os.path.join(root, file))
        stock_code_short, stock_names = check_stock_file(os.path.join(root, file))

        if (stock_code_short not in unique_stock_codes):
          unique_stock_codes.append(stock_code_short)
        unique_stock_names.extend(stock_names)

  unique_stock_codes.sort()
  print(unique_stock_codes)
  unique_stock_names.sort()
  print(unique_stock_names)

# 载入股票数据
def load_stock_file(stock_file_path):
  stock_code = os.path.basename(stock_file_path).split('.')[0]
  df = pd.read_csv(stock_file_path, encoding = 'gbk')
  df['Date'] = pd.to_datetime(df['交易日期'])
  # 将日期列设置为索引
  df.set_index('Date', inplace=True)
  # 按照日期升序排列
  df.sort_index(ascending=True, inplace=True)

  return stock_code, df

# 标准化数据
def normalize_stock_data(stock_code, df):
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
    return stock_code, stock_code_prefix_value, df
  
  # 2. '股票名称'
  stock_previous_name = ''
  df['name_std'] = STOCK_NAME_PREFIXES_NORMAL
  for row in df.itertuples():
    stock_name, stock_name_prefixe, stock_name_prefixe_value = parse_stock_name(row.股票名称)
    # 创业板 / 科创板：第 6 个交易日起涨跌幅限制为 20%
    # 将涨跌幅调整的为10%以内
    if (stock_code_prefix_value == 5) and (stock_code_prefix_value == 10) :
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
    df.at[row.Index, 'name_std'] = stock_name_prefixe_value
    #print(stock_name, stock_name_prefixe, stock_name_prefixe_value)

  # 3. '开盘价', '最高价', '最低价', '收盘价'
  # 3.1 对价格类特征（open/close）使用 Z-score 标准化
  # 3.1.1 合并两个字段的数据，计算共同的均值和标准差
  combined_data = np.hstack((
      df['开盘价'].values.reshape(-1, 1),
      df['最高价'].values.reshape(-1, 1),
      df['最低价'].values.reshape(-1, 1),
      df['收盘价'].values.reshape(-1, 1)
  )).flatten()  # 合并为一维数组
  mean = np.mean(combined_data)
  std = np.std(combined_data)
  # 3.1.2 对两个字段分别应用同一均值和标准差进行标准化
  df['open_std']  = (df['开盘价'] - mean) / std
  df['high_std']  = (df['最高价'] - mean) / std
  df['low_std']   = (df['最低价'] - mean) / std
  df['close_std'] = (df['收盘价'] - mean) / std

  # 4. '成交量'
  # 4.1 对成交量（volume）使用对数变换
  df['volume_log'] = np.log1p(df['成交量'])
  # 4.2 对成交量（volume_log）使用Z-score 标准化
  scaler = StandardScaler()
  df['volume_log_std'] = scaler.fit_transform(df[['volume_log']])

  # 5. '换手率'
  # 5.1 对换手率（turnover）使用 Z-score 标准化
  scaler = StandardScaler()
  # 5.1 对换手率（turnover）使用 Min-Max 标准化，换手率范围通常0-100%，缩放到0-1
  #scaler = MinMaxScaler(feature_range=(0, 1))
  df['turnover_std'] = scaler.fit_transform(df[['换手率']])

  # 6. '流通市值', '总市值'
  # 6.1 基于流通市值、总市值计算流通股数和总股数
  df['outstanding_shares'] = df['流通市值'] / df['收盘价']
  df['total_shares'] = df['总市值'] / df['收盘价']
  # 6.2 对流通股数、总股数使用分段平滑 Z-score
  df['outstanding_shares_std'] = zscore_based_smoothing(df['outstanding_shares'], 3)
  df['total_shares_std'] = zscore_based_smoothing(df['total_shares'], 3)
  # 6.3 对计算流通比，并使用分段平滑 Z-score
  df['outstanding_shares_ratio'] = df['outstanding_shares'] / df['total_shares']
  df['outstanding_shares_ratio_std'] = zscore_based_smoothing(df['outstanding_shares_ratio'], 3)
  # 6.4 对流通股数、总股数使用对数变换
  df['outstanding_shares_log'] = np.log1p(df['outstanding_shares_std'])
  df['total_shares_log'] = np.log1p(df['total_shares_std'])
  # 6.5 合并对数变换后的流通股数、总股数的数据，计算共同的均值和标准差
  combined_data = np.hstack((
      df['outstanding_shares_log'].values.reshape(-1, 1),
      df['total_shares_log'].values.reshape(-1, 1)
  )).flatten()  # 合并为一维数组
  mean = np.mean(combined_data)
  std = np.std(combined_data)
  # 6.6 对两个字段分别应用同一均值和标准差进行标准化
  df['outstanding_shares_log_std'] = (df['outstanding_shares_log'] - mean) / std
  df['total_shares_log_std'] = (df['total_shares_log'] - mean) / std

  # 7. '市盈率TTM', '市销率TTM', '市现率TTM', '市净率'
  # 7.1 对'市盈率TTM', '市销率TTM', '市现率TTM', '市净率'使用 Z-score 标准化
  for col in ['市盈率TTM', '市销率TTM', '市现率TTM', '市净率']:
    df[col] = df[col].replace([np.nan, np.inf, -np.inf], 0)
  scaler = StandardScaler()
  df['pe_std'] = scaler.fit_transform(df[['市盈率TTM']])
  df['ps_std'] = scaler.fit_transform(df[['市销率TTM']])
  df['pcf_std'] = scaler.fit_transform(df[['市现率TTM']])
  df['pb_std'] = scaler.fit_transform(df[['市净率']])

  # 8. '量比'
  # 8.1 对'量比'使用 Z-score
  scaler = StandardScaler()
  df['volume_ratio_std'] = scaler.fit_transform(df[['量比']])

  # 9. '涨跌幅', '振幅'
  # 9.1 对'涨跌幅', '振幅'使用 Z-score 标准化
  scaler = StandardScaler()
  df['change_ratio_std'] = scaler.fit_transform(df[['涨跌幅']])
  df['change_ratio_range_std'] = scaler.fit_transform(df[['振幅']])

  # 'name_std'
  # 'open_std', 'high_std', 'low_std', 'close_std'
  # 'volume_log_std', 'turnover_std',
  # 'outstanding_shares_ratio_std', 'outstanding_shares_log_std', 'total_shares_log_std'
  # 'pe_std', 'ps_std', 'pcf_std', 'pb_std'
  # 'volume_ratio_std', 'change_ratio_std'
  return stock_code, stock_code_prefix_value, df

# 检查数据处理的正确与否
def print_stock_data(stock_code, df, figure_path=None):
  # 1. 绘制 K 线图、成交量和成交额图
  features = df[['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额']]
  features = features.copy()
  # 重命名列名以符合 mplfinance 默认规则
  features.rename(columns={'开盘价': 'Open', '最高价': 'High', '最低价': 'Low', '收盘价': 'Close', '成交量': 'Volume', '成交额': 'Turnvolume'}, inplace=True)
  figure_file_path = os.path.join(figure_path, f"{stock_code}.png") if figure_path is not None else None
  plot_stock(features, stock_code, figure_file_path)

  # 2. 绘制特征的折线图
  # 2.1 价格
  features = df[['开盘价', '最高价', '最低价', '收盘价', 'open_std', 'high_std', 'low_std', 'close_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.prices.png") if figure_path is not None else None
  plot_lines(features, ['开盘价', '最高价', '最低价', '收盘价'], ['open_std', 'high_std', 'low_std', 'close_std'], figure_file_path)

  # 2.1.2 价格 vs 成交量
  features = df[['收盘价', '成交量']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.prices.vs.volume.png") if figure_path is not None else None
  plot_lines(features, ['收盘价'], ['成交量'], figure_file_path)

  # 2.1.3 价格 vs 换手率
  features = df[['收盘价', '换手率']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.prices.vs.turnover.png") if figure_path is not None else None
  plot_lines(features, ['收盘价'], ['换手率'], figure_file_path)

  # 2.2 成交量
  features = df[['成交量', 'volume_log_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.volume.png") if figure_path is not None else None
  plot_lines(features, ['成交量'], ['volume_log_std'], figure_file_path)

  # 2.3 换手率
  features = df[['换手率', 'turnover_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.turnover.png") if figure_path is not None else None
  plot_lines(features, ['换手率'], ['turnover_std'], figure_file_path)

  # 2.4 流通股数, 总股数, 流通比
  features = df[['outstanding_shares', 'total_shares', 'outstanding_shares_log_std', 'total_shares_log_std', 'outstanding_shares_ratio_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.shares.png") if figure_path is not None else None
  plot_lines(features, ['outstanding_shares', 'total_shares'], ['outstanding_shares_log_std', 'total_shares_log_std', 'outstanding_shares_ratio_std'], figure_file_path)

  # 2.6 '市盈率TTM', '市销率TTM', '市现率TTM', '市净率'
  features = df[['市盈率TTM', '市销率TTM', '市现率TTM', '市净率', 'pe_std', 'ps_std', 'pcf_std', 'pb_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.ttms.png") if figure_path is not None else None
  plot_lines(features, ['市盈率TTM', '市销率TTM', '市现率TTM', '市净率'], ['pe_std', 'ps_std', 'pcf_std', 'pb_std'], figure_file_path)

  # 2.7 '量比'
  features = df[['量比', 'volume_ratio_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.volume.ratio.png") if figure_path is not None else None
  plot_lines(features, ['量比'], ['volume_ratio_std'], figure_file_path)

  # 2.8 '涨跌幅'
  features = df[['涨跌幅', '振幅', 'change_ratio_std', 'change_ratio_range_std']]
  features = features.copy()
  figure_file_path = os.path.join(figure_path, f"{stock_code}.change.png") if figure_path is not None else None
  plot_lines(features, ['涨跌幅', '振幅'], ['change_ratio_std', 'change_ratio_range_std'], figure_file_path)
