# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 23:22:08 2025

@author: 27089
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 设置中文显示（如果标签需要中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统支持的字体
plt.rcParams['axes.unicode_minus'] = False

# 示例数据（模型名称和对应样本量）
models = ['GPT-4', 'PaLM 2', 'LLaMA 2', 'Claude 2', 'GPT-3.5']
sample_sizes = [13000, 7800, 6500, 4200, 3500]  # 单位：亿token

# 颜色设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('大语言模型训练数据量对比', fontsize=14, pad=20)
ax.set_xlabel('模型名称', fontsize=12)
ax.set_ylabel('训练数据量 (亿token)', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 设置y轴范围（留出顶部空间）
ax.set_ylim(0, max(sample_sizes) * 1.2)

# 初始空柱状图
bars = ax.bar(models, [0]*len(models), color=colors, width=0.6)

# 添加数值标签函数
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}亿',
                ha='center', va='bottom', fontsize=10)

# 动画更新函数
def update(frame):
    # 当前应该显示到第几个柱子（0-based）
    current_bar = frame // 10  # 每个柱子用10帧动画
    
    if current_bar < len(models):
        # 更新当前柱子的高度（线性增长）
        progress = (frame % 10 + 1) / 10
        bars[current_bar].set_height(sample_sizes[current_bar] * progress)
        
        # 如果是最后一个柱子，添加所有标签
        if current_bar == len(models) - 1 and frame % 10 == 9:
            add_labels(bars)
    
    return bars

# 创建动画
ani = animation.FuncAnimation(
    fig, update, frames=len(models)*10,  # 每个柱子10帧
    interval=100, blit=False, repeat=False)

# 添加脚注
plt.figtext(0.5, 0.01, '数据来源: 各模型公开技术报告 (2023)', 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.show()

# 如需保存动画（取消注释以下代码）
ani.save('model_comparison.gif', writer='pillow', fps=10, dpi=100)