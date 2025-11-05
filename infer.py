import torch
from glob import glob
import os 
import numpy as np
import cv2
from NeuFlow.neuflow import NeuFlow 
from NeuFlow.backbone_v7 import ConvBlock 
from data_utils import flow_viz 

image_width = 768
image_height = 432

def get_cuda_image(image_path):
    """读取图像并转换为CUDA上的半精度张量"""
    image = cv2.imread(image_path) 
    image = cv2.resize(image, (image_width, image_height)) 
    image = torch.from_numpy(image).permute(2, 0, 1).half()  # 转换为张量并调整维度顺序，转为半精度
    return image[None].cuda() 
def fuse_conv_and_bn(conv, bn):
    fusedconv = (
        torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False) 
        .to(conv.weight.device)
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)  # 展平卷积权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # 计算BN权重
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))  # 复制融合后的权重
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))  # 计算BN偏置
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # 复制融合后的偏置
    return fusedconv

image_path_list = sorted(glob('test_images/*.jpg'))
vis_path = 'test_results/' 
device = torch.device('cuda') 

model = NeuFlow().to(device)
checkpoint = torch.load('neuflow_mixed.pth', map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=True)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # 融合第一个卷积和BN层
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # 融合第二个卷积和BN层
        delattr(m, "norm1")  # 删除批归一化层属性
        delattr(m, "norm2")  # 删除批归一化层属性
        m.forward = m.forward_fuse  # 更新前向传播方法

model.eval()  # 设置模型为评估模式
model.half()  # 使用半精度浮点数
model.init_bhwd(1, image_height, image_width, 'cuda')  # 初始化模型输入尺寸

# 创建结果保存目录
if not os.path.exists(vis_path):
    os.makedirs(vis_path)

# 处理连续的图像对
for image_path_0, image_path_1 in zip(image_path_list[:-1], image_path_list[1:]):
    print(image_path_0)  # 打印当前处理的图像路径
    image_0 = get_cuda_image(image_path_0)  # 读取第一张图像到CUDA
    image_1 = get_cuda_image(image_path_1)  # 读取第二张图像到CUDA
    file_name = os.path.basename(image_path_0)  # 获取文件名
    with torch.no_grad(): 
        flow = model(image_0, image_1)[-1][0]  # 计算光流，取最后一层输出
        flow = flow.permute(1,2,0).cpu().numpy()  # 调整维度顺序并转换为numpy数组
        flow = flow_viz.flow_to_image(flow)  # 将光流转换为彩色图像
        #image_0 = cv2.resize(cv2.imread(image_path_0), (image_width, image_height))
        #cv2.imwrite(vis_path + file_name, np.vstack([image_0, flow])) 