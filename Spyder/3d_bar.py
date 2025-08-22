import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 示例数据
models = ['GPT-4', 'PaLM 2', 'LLaMA 2', 'Claude 2', 'GPT-3.5']
sample_sizes = [13000, 7800, 6500, 4200, 3500]  # 单位：亿token
positions = np.arange(len(models))  # 模型的位置

# 颜色设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 创建3D图形
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('大语言模型训练数据量对比 (3D)', fontsize=14, pad=20)
ax.set_xlabel('模型名称', fontsize=12, labelpad=10)
ax.set_ylabel('', fontsize=12)
ax.set_zlabel('训练数据量 (亿token)', fontsize=12, labelpad=10)

# 设置视角
ax.view_init(elev=10, azim=-60)

# 设置坐标轴范围
ax.set_xlim(-0.5, len(models)-0.5)
ax.set_ylim(-0.5, 0.5)  # 固定Y轴范围
ax.set_zlim(0, max(sample_sizes) * 1.3)

# 设置X轴刻度
ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_yticks([])  # 隐藏Y轴刻度

# 预先创建所有柱体（高度为0）
bars = []
labels = []
for i in range(len(models)):
    # 创建一个长方体
    bar = ax.bar3d(
        positions[i]-0.15, -0.15, 0,  # 左下角坐标(x, y, z)
        0.6, 0.2, 0,               # 柱体尺寸 (dx, dy, dz)
        color=colors[i],
        alpha=0.7,
        shade=True,
        edgecolor='w',
        linewidth=0.5
    )
    bars.append(bar)
    
    # 创建标签（初始隐藏）
    label = ax.text(positions[i], 0.5, 0, '', 
                    ha='center', va='bottom', 
                    fontsize=9, color='k', zorder=10, visible=False)
    labels.append(label)

# 动画更新函数（逐个更新柱体）
def update(frame):
    # 当前应该显示到第几个柱子（0-based）
    current_bar = frame // 10
    
    updated_artists = []
    
    for i in range(len(models)):
        if i < current_bar:
            # 已完成的柱子 - 使用完整高度
            height = sample_sizes[i]
            alpha = 0.7
        elif i == current_bar:
            # 当前柱子增长动画
            progress = (frame % 10 + 1) / 10
            height = sample_sizes[i] * progress
            alpha = 1.0  # 当前柱子高亮
        else:
            # 未开始的柱子 - 高度为0
            height = 0
            alpha = 0.0
            
        # 创建长方体新顶点
        x, y, z = positions[i]-0.15, -0.15, 0
        dx, dy, dz = 0.6, 0.2, height
        
        verts = np.array([
            # 底部
            [[x, y, z], [x, y+dy, z], [x+dx, y+dy, z], [x+dx, y, z]],
            # 顶部
            [[x, y, z+dz], [x, y+dy, z+dz], [x+dx, y+dy, z+dz], [x+dx, y, z+dz]],
            # 左面
            [[x, y, z], [x, y, z+dz], [x, y+dy, z+dz], [x, y+dy, z]],
            # 右面
            [[x+dx, y, z], [x+dx, y+dy, z], [x+dx, y+dy, z+dz], [x+dx, y, z+dz]],
            # 前面
            [[x, y, z], [x+dx, y, z], [x+dx, y, z+dz], [x, y, z+dz]],
            # 后面
            [[x, y+dy, z], [x, y+dy, z+dz], [x+dx, y+dy, z+dz], [x+dx, y+dy, z]],
        ])
        
        # 更新柱体
        bars[i].set_verts(verts)
        bars[i].set_alpha(alpha)
        updated_artists.append(bars[i])
        
        # 更新标签
        if height > 0:
            labels[i].set(position=(positions[i]+0.1, -0.3, height * 1.05))
            labels[i].set(text=f'{int(height)}亿')
            labels[i].set(visible=True)
            updated_artists.append(labels[i])
        else:
            labels[i].set(visible=False)
    
    return updated_artists

# 创建动画
ani = animation.FuncAnimation(
    fig, update, 
    frames=len(models)*10 + 5,  # 每个柱子10帧，结束时多停留5帧
    interval=100, 
    blit=True,  # 启用blit优化性能
    repeat=False
)

# 添加脚注
plt.figtext(0.5, 0.01, '数据来源: 各模型公开技术报告 (2023)', 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()

# 保存动画
ani.save('model_comparison_3d.gif', writer='pillow', fps=10, dpi=120)
print("动画已保存为 'model_comparison_3d.gif'")
plt.show()