import matplotlib.pyplot as plt

def visualize_enhancement(images, layout, titles):
    """
    绘制对应layout的拼接图，一般用于观察图像增强效果
    
    参数:
    images : list of ndarray 
        图像列表，第一个应为原始图像，后续为增强结果
    layout : tuple 
        子图布局 (rows, cols)
    titles : list of str 
        每个子图的标题列表，长度必须等于图像数量
    """
    # 创建子图画布
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, 
                           figsize=(4*cols, 4*rows))  # 根据布局自动调整画布大小
    
    # 展平轴数组便于遍历
    axes = axes.ravel() if rows * cols > 1 else [axes]
    
    # 遍历所有图像绘制子图
    for ax, img, title in zip(axes, images, titles): 
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=10)  
       # ax.axis('off')  # 关闭坐标轴
    
    plt.tight_layout()
    return fig, axes
