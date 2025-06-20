import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

class TreeCanopyEnhancer:
    """
    树冠图像增强处理器，支持颜色、对比度、纹理调整
    --------------------------------------------
    通过参数化调节实现以下功能：
    - 颜色增强：全局/绿色区域饱和度调整、自适应白平衡
    - 对比度增强：CLAHE直方图均衡、直方图拉伸
    - 纹理增强：锐化滤波与导向滤波融合
    - 曝光调节：gamma校正与自适应直方图均衡
    处理方法：
    process() 依次执行对比度增强 -> 颜色增强 -> 纹理增强流程，
    _enhance_xxx 系列方法可各环节独立调用，
    _exposure_xxx/_saturation_xxx 提供额外曝光补偿功能
    """
    def __init__(self,
                 # 颜色增强参数
                 saturation_factor=1.2,
                 green_hue_range=(30, 90),
                 green_saturation_scale=1.5,
                 green_mask_weight=0.5,
                 green_merge_gamma=0,
                 
                 # 纹理增强参数
                 sharp_kernel=np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]]),
                 sharp_weight=0.7,
                 guided_weight=0.3):
        
        # 初始化颜色增强参数
        self.saturation_factor = saturation_factor
        self.green_hue_range = green_hue_range
        self.green_saturation_scale = green_saturation_scale
        self.green_mask_weight = green_mask_weight
        self.green_merge_gamma = green_merge_gamma
        
        # 初始化纹理增强参数
        self.sharp_kernel = sharp_kernel
        self.sharp_weight = sharp_weight
        self.guided_weight = guided_weight

    def _check_value_range(self, img):
        """确保图像数值范围在0-255之间"""
        if img.dtype == np.float32 or img.dtype == np.float64:
            if np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
        return img.astype(np.uint8)

    def _enhance_contrast(self, img, clahe_clip_limit=3, 
                          clahe_tile_size=(2,2)):
        """使用CLAHE增强对比度"""
        # 对明度图层进行单独的区域直方图均衡
         # 初始化对比度增强参数
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _rescale_histgram(self, img, hist_percentiles=(5,95)):
        """使用直方图拉伸对比度"""
        # 全局直方图拉伸
        p_low, p_high = np.percentile(img, hist_percentiles)
        return exposure.rescale_intensity(img, in_range=(p_low, p_high))

    def _enhance_texture(self, img, radius=3, eps=200):
        """可调参数纹理增强"""
        # 锐化滤波
        sharpened = cv2.filter2D(img, -1, self.sharp_kernel)
        
        # 导向滤波参数设置
        guided = cv2.ximgproc.guidedFilter(
            guide=img,
            src=sharpened,
            radius=radius,
            eps=eps
        )
        
        # 加权融合
        return cv2.addWeighted(
            sharpened, self.sharp_weight,
            guided, self.guided_weight, 0
        )


    def _enhance_color(self, img):
        """可调参数颜色增强"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 全局饱和度增强
        hsv[:,:,1] = cv2.multiply(
            hsv[:,:,1], 
            self.saturation_factor
        )
        
        # 绿色区域选择增强
        green_mask = cv2.inRange(
            hsv[:,:,0], 
            self.green_hue_range[0],
            self.green_hue_range[1]
        )
        hsv[:,:,1] = cv2.addWeighted(
            hsv[:,:,1], 
            self.green_saturation_scale,
            green_mask, 
            self.green_mask_weight, 
            self.green_merge_gamma
        )
        
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return cv2.xphoto.createSimpleWB().balanceWhite(enhanced)
    
    def _exposure_enhanced(self, img, gamma):
        """ 调整gamma,<1图像变亮"""
        return exposure.adjust_gamma(img, gamma)
    
    def _exposure_clahe(self, img, kernel_size=(4,4), clip_limit=0.1):
        return exposure.equalize_adapthist(img, kernel_size, clip_limit)
    
    def _saturation_enhanced(self, img, gain=1.2):
        """全局饱和度增强"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = cv2.multiply(
            hsv[:,:,1], 
            gain
        )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def process(self, img):
        img = self._check_value_range(img)
        img = self._enhance_contrast(img)
        img = self._enhance_color(img)
        img = self._enhance_texture(img)
        return img

#import matplotlib.pyplot as plt
#import numpy as np

# 已整合至visualize_tools.py，后期去除
def visualize_enhancement(images, layout, titles):
    """
    可视化增强结果对比
    
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

if __name__ == "__main__":
    basic_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])
    
    custom_enhancer = TreeCanopyEnhancer(
        # 颜色参数
        saturation_factor=1.5,
        green_hue_range=(25, 90),
        green_saturation_scale=1.5,
        green_mask_weight=0.7,
        green_merge_gamma=20,
        
        # 纹理参数
        sharp_kernel= basic_kernel,
        sharp_weight=0.5,
        guided_weight=0.5
    )

    img = cv2.imread("./test_img/test_hy.png")
    enhanced_texture = custom_enhancer._enhance_texture(img) # 纹理增强
    enhanced_hist = custom_enhancer._rescale_histgram(img, hist_percentiles=(2,98)) # 直方图均衡
    enhanced_clahe = custom_enhancer._enhance_contrast(img, clahe_tile_size=(2,2), # CLAHE均衡
                                                       clahe_clip_limit=1)
    enhanced_saturation = custom_enhancer._saturation_enhanced(img, gain=1.5) # 饱和度增强
    enhanced_exposure = custom_enhancer._exposure_enhanced(img, gamma=0.8) # 提高曝光度

    #cv2.imshow("hist", enhanced_hist)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    fig, axes = visualize_enhancement([img, enhanced_texture,
                           enhanced_hist, enhanced_clahe, 
                           enhanced_saturation, enhanced_exposure],
                          (3,2),
                          ["original", "texture", 
                           "histgram", "CLAHE",
                           "saturation", "exposure"])
    plt.show()
    