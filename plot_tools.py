import os
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns


def plot(df, features, file_path=None):
  # 设置图片清晰度
  plt.rcParams['figure.dpi'] = 300
  # 设置支持中文的字体，这里以 Arial Unicode MS 为例
  plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
  # 解决负号显示问题
  plt.rcParams['axes.unicode_minus'] = False

  # 确定所有特征的最小值和最大值
  min_val = df[features].min().min()
  max_val = df[features].max().max()
  # 手动设置 bins
  num_bins = 30
  bins = np.linspace(min_val, max_val, num_bins + 1)

  feature_labels = '/'.join(features)
  plt.figure(figsize=(18, 6))

  # 1. 绘制直方图
  plt.subplot(1, 3, 1)
  for feature in features:
      sns.histplot(df[feature], bins=bins, kde=False, label=feature, alpha=0.5)
  plt.title(f"{feature_labels}的直方图")
  plt.xlabel("特征值")
  plt.ylabel('频数')
  plt.legend()

  # 2. 绘制核密度估计图
  plt.subplot(1, 3, 2)
  for feature in features:
      sns.kdeplot(df[feature], fill=True, label=feature)
  plt.title(f"{feature_labels}的核密度估计图")
  plt.xlabel("特征值")
  plt.ylabel('密度')
  plt.legend()

  # 3. 绘制箱线图
  plt.subplot(1, 3, 3)
  # 将数据转换为长格式
  df_melted = pd.melt(df, value_vars=features)
  sns.boxplot(x='variable', y='value', data=df_melted)
  plt.title(f"{feature_labels}的箱线图")
  plt.xlabel("特征")
  plt.ylabel("特征值")
  
  plt.tight_layout()
  if file_path:
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
  else:
    plt.show()

def plot_lines(df, features1, features2=[], file_path=None):
  # 设置图片清晰度
  plt.rcParams['figure.dpi'] = 300
  # 设置支持中文的字体，这里以 Arial Unicode MS 为例
  plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
  # 解决负号显示问题
  plt.rcParams['axes.unicode_minus'] = False

  feature_labels = '/'.join(features1 + features2)
  # 创建图形和主坐标轴对象
  fig, ax1 = plt.subplots(figsize=(18, 6))

  # 绘制 features1 中的特征曲线
  color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
  for i, feature in enumerate(features1):
      color = color_list[i % len(color_list)]
      ax1.plot(df.index, df[feature], color=color, label=feature)
  #ax1.set_xlabel('Index')
  ax1.set_ylabel(', '.join(features1))
  ax1.tick_params(axis='y')
  ax1.legend(loc='upper left')

  if features2 != []:        
      # 创建第二个 y 坐标轴
      ax2 = ax1.twinx()

      # 绘制 features2 中的特征曲线
      for i, feature in enumerate(features2):
          color = color_list[(i + len(features1)) % len(color_list)]
          ax2.plot(df.index, df[feature], color=color, label=feature)
      ax2.set_ylabel(', '.join(features2))
      ax2.tick_params(axis='y')
      ax2.legend(loc='upper right')

  # 设置图表标题
  plt.title(f"{feature_labels}直线图")

  if file_path:
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
  else:
    plt.show()

def plot_stock(df, stock_code='', file_path=None):
  # 设置图片清晰度
  plt.rcParams['figure.dpi'] = 300
  # 设置支持中文的字体，这里以 Arial Unicode MS 为例
  plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
  # 解决负号显示问题
  plt.rcParams['axes.unicode_minus'] = False

  market_colors = mpf.make_marketcolors(
    up = 'green', down = 'red', edge = 'black',
    wick = { 'up' : 'blue', 'down' : 'orange' },
    volume = 'purple',
    ohlc = 'black')
  style = mpf.make_mpf_style(marketcolors=market_colors, rc={'font.family': 'Arial Unicode MS'})

  addplot = [
    mpf.make_addplot(df['Turnvolume'], panel=2, ylabel='成交额'),
    mpf.make_addplot(df['Volume'], panel=1, ylabel='成交量')
  ]
  # 绘制 K 线图、成交量和成交额图
  fig, axes = mpf.plot(
    df,
    type='candle',
    volume=False,
    mav=(5, 8, 13),
    show_nontrading=False, 
    addplot=addplot,
    figsize=(18, 6),
    style=style,
    xrotation=0, datetime_format='%Y-%m-%d',
    warn_too_much_data=len(df) + 1,
    returnfig=True)
  
  # 设置标题
  axes[0].set_title(f"{stock_code}股票走势")
  axes[0].set_xlabel('日期')
  axes[0].set_ylabel('价格')
      
  if file_path:
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
  else:
    plt.show()
