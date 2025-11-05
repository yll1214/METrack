import numpy as np


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

import numpy as np
# 需要在文件开头添加OpenCV导入
import cv2


def flow_to_vectors(flow_image):
    """
    将光流图像转换为光流向量场
    
    Args:
        flow_image: 光流图像 (H, W, 3)，Middlebury颜色编码
    
    Returns:
        flow_vectors: 光流向量场 (H, W, 2)，包含(x, y)位移向量
    """
    if flow_image is None:
        return None
        
    h, w = flow_image.shape[:2]
    flow_vectors = np.zeros((h, w, 2), dtype=np.float32)
    
    # 将Middlebury颜色编码转换回光流向量
    # Middlebury颜色编码: 色调表示方向，饱和度表示幅度
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(flow_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # 将色调转换为角度（0-360度）
    angle = hue * 2.0  # 因为OpenCV中Hue的范围是0-180
    
    # 将饱和度转换为幅度（0-1范围）
    magnitude = saturation / 255.0
    
    # 计算最大幅度用于归一化（需要根据具体的光流范围调整）
    max_magnitude = 20.0  # 这个值可能需要根据实际光流范围调整
    
    # 计算实际的光流向量
    angle_rad = np.deg2rad(angle)
    flow_vectors[:, :, 0] = magnitude * max_magnitude * np.cos(angle_rad)  # u分量
    flow_vectors[:, :, 1] = magnitude * max_magnitude * np.sin(angle_rad)  # v分量
    
    return flow_vectors


def flow_to_vectors_simple(flow_image, max_flow=20.0):
    """
    简化的光流图像到向量转换（基于颜色值直接估计）
    
    Args:
        flow_image: 光流图像 (H, W, 3)
        max_flow: 最大光流值估计
    
    Returns:
        flow_vectors: 光流向量场 (H, W, 2)
    """
    if flow_image is None:
        return None
        
    h, w = flow_image.shape[:2]
    flow_vectors = np.zeros((h, w, 2), dtype=np.float32)
    
    # 简化的转换方法：基于RGB值估计光流
    # R通道可能表示水平方向，G通道可能表示垂直方向
    # 需要根据实际的光流可视化编码方式调整
    
    # 假设光流图像是标准Middlebury编码，我们需要反向计算
    # 这里提供一个简化的近似方法
    
    # 转换为灰度图来估计幅度
    gray = cv2.cvtColor(flow_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    magnitude = gray / 255.0 * max_flow
    
    # 从颜色估计方向
    normalized = flow_image.astype(np.float32) / 255.0
    r, g, b = normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]
    
    # 简化的方向估计（这只是一个近似）
    angle = np.arctan2(g - 0.5, r - 0.5)
    
    flow_vectors[:, :, 0] = magnitude * np.cos(angle)
    flow_vectors[:, :, 1] = magnitude * np.sin(angle)
    
    return flow_vectors


def get_flow_magnitude_direction(flow_image):
    """
    从光流图像提取幅度和方向信息
    
    Args:
        flow_image: 光流图像 (H, W, 3)
    
    Returns:
        magnitude: 光流幅度图 (H, W)
        direction: 光流方向图 (H, W)，角度（弧度）
    """
    if flow_image is None:
        return None, None
        
    # 使用更可靠的方法：通过分析颜色特征来估计
    hsv = cv2.cvtColor(flow_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # 色调表示方向（0-360度）
    direction = np.deg2rad(hue * 2.0)  # OpenCV中Hue范围是0-180
    
    # 饱和度表示幅度（归一化到0-1）
    magnitude = sat / 255.0
    
    return magnitude, direction


def extract_roi_flow(flow_vectors, bbox):
    """
    从光流向量场中提取感兴趣区域的光流统计信息
    
    Args:
        flow_vectors: 光流向量场 (H, W, 2)
        bbox: 边界框 [x, y, w, h]
    
    Returns:
        mean_flow: 平均光流向量 [mean_u, mean_v]
        median_flow: 中值光流向量 [median_u, median_v]
    """
    if flow_vectors is None:
        return [0, 0], [0, 0]
    
    x, y, w, h = map(int, bbox)
    x = max(0, min(x, flow_vectors.shape[1] - 1))
    y = max(0, min(y, flow_vectors.shape[0] - 1))
    w = max(1, min(w, flow_vectors.shape[1] - x))
    h = max(1, min(h, flow_vectors.shape[0] - y))
    
    # 提取ROI区域的光流
    roi_flow = flow_vectors[y:y+h, x:x+w, :]
    
    if roi_flow.size == 0:
        return [0, 0], [0, 0]
    
    # 计算统计信息
    roi_flow_flat = roi_flow.reshape(-1, 2)
    
    # 移除异常值（基于幅度过滤）
    magnitudes = np.linalg.norm(roi_flow_flat, axis=1)
    valid_mask = magnitudes < np.percentile(magnitudes, 95)  # 去除前5%的异常值
    
    if np.any(valid_mask):
        valid_flow = roi_flow_flat[valid_mask]
        mean_flow = np.mean(valid_flow, axis=0)
        median_flow = np.median(valid_flow, axis=0)
    else:
        mean_flow = np.mean(roi_flow_flat, axis=0)
        median_flow = np.median(roi_flow_flat, axis=0)
    
    return mean_flow, median_flow




def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel
    

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_vis_flow_tofile(flow, output_path):
    vis_flow = flow_to_image(flow)
    from PIL import Image
    img = Image.fromarray(vis_flow)
    img.save(output_path)