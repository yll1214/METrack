import torch
from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target, grounding_resize
from copy import deepcopy
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.METrack.METrack import build_model
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_cxcywh_to_xywh, box_cxcywh_to_xyxy
import numpy as np
import matplotlib.pyplot as plt
from lib.test.utils.hann import hann2d
from scipy import signal
from transformers import BertModel
from transformers import BertTokenizer
from lib.utils.misc import NestedTensor

class METrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(METrack, self).__init__(params)
        network = build_model(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'],
                                strict=False)
        self.map_size = params.search_size // 16
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        
        # ==================== 调试和可视化参数 ====================
        self.debug = self.params.debug
        self.frame_id = 0
        
        # ==================== 阈值参数集中设置 ====================
        self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
        self.update_threshold = getattr(self.cfg.TEST, 'UPDATE_THRESHOLD', 0.85)
        self.redetection_threshold = getattr(self.cfg.TEST, 'REDETECTION_THRESHOLD', 0.3)
        self.low_confidence_threshold = getattr(self.cfg.TEST, 'LOW_CONFIDENCE_THRESHOLD', 0.5)
        
        # ==================== 相关滤波参数 ====================
        self.cf_enabled = True
        self.cf_lr = getattr(self.cfg.TEST, 'CF_LR', 0.02)
        self.cf_padding = getattr(self.cfg.TEST, 'CF_PADDING', 1.5)
        self.cf_output_sigma_factor = getattr(self.cfg.TEST, 'CF_OUTPUT_SIGMA_FACTOR', 0.1)
        self.cf_template = None
        self.cf_model = None
        
        # ==================== 模板库参数 ====================
        self.template_library = {}
        self.max_templates_per_sequence = getattr(self.cfg.TEST, 'MAX_TEMPLATES', 5)
        self.current_sequence_id = None

        self.prev_bearing = None
        self.bearing_rate_filter = None
        self.bearing_rate_smoothing = 0.3
        
        # ==================== 光流参数 ====================
        self.use_optical_flow = True
        self.flow_smoothing_factor = 0.3
        self.prev_gray = None
        self.prev_bbox = None
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # ==================== 可视化颜色设置 ====================
        self.colors = {
            'tracker': (0, 255, 0),
            'flow': (255, 0, 0),
            'fused': (0, 0, 255),
            'center': (255, 255, 0),
            'trajectory': (0, 255, 255),
            'gt': (255, 0, 255),
            'cf': (0, 165, 255)
        }

        # ==================== 轨迹可视化相关参数 ====================
        self.trajectory = []
        self.trajectory_length = 50
        self.trajectory_thickness = 2

        # ==================== 其他参数 ====================
        self.feat_size = self.params.search_size // 16
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.MODEL.BACKBONE.LANGUAGE.VOCAB_PATH, do_lower_case=True)
        self.threshold = self.cfg.TEST.THRESHOLD
        self.has_cont = self.cfg.TRAIN.CONT_WEIGHT > 0
        self.max_score = 0

        self.sequence_index = 0
        self.sequence_id = 1
        self.sequence_ids = list(range(1, 1001))
        self.frame_counter = 1
        
        # 初始化debug目录
        self._init_debug_dir()
        
    def _init_debug_dir(self):
        """初始化debug输出目录"""
        base_dir = "debug_results"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        self.sequence_save_dir = os.path.join(base_dir, f"sequence_{self.sequence_id}")
        os.makedirs(self.sequence_save_dir, exist_ok=True)
        
    def _create_cf_hanning_window(self, size):
        """创建汉宁窗"""
        return np.outer(np.hanning(size[0]), np.hanning(size[1]))
    
    def _create_cf_gaussian_label(self, size, sigma):
        """创建高斯标签"""
        sz = np.array(size)
        rs, cs = np.meshgrid(np.arange(sz[0]) - np.floor(sz[0]/2),
                           np.arange(sz[1]) - np.floor(sz[1]/2))
        labels = np.exp(-0.5 / sigma**2 * (rs**2 + cs**2))
        return labels
    
    def _train_cf(self, x, y):
        """训练相关滤波"""
        alphaf_list = []
        for i in range(x.shape[2]):
            k = self._linear_correlation(x[:, :, i], x[:, :, i])
            alphaf = np.fft.fft2(y) / (np.fft.fft2(k) + 1e-8)
            alphaf_list.append(alphaf)
        return np.mean(alphaf_list, axis=0)
    
    def _linear_correlation(self, x1, x2):
        """线性相关"""
        return signal.fftconvolve(x1, x2[::-1, ::-1], mode='same')
    
    def _extract_cf_features(self, image_patch):
        """提取相关滤波特征"""
        if len(image_patch.shape) == 3:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_patch
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        features = np.stack([magnitude, orientation], axis=2)
        features = features.astype(np.float32)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def initialize_cf(self, image, bbox):
        """初始化相关滤波"""
        if not self.cf_enabled:
            return
            
        try:
            x, y, w, h = [int(coord) for coord in bbox]
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            
            template_patch = image[y:y+h, x:x+w]
            
            if template_patch.size == 0:
                return
                
            target_size = (64, 64)
            template_patch = cv2.resize(template_patch, target_size)
            
            features = self._extract_cf_features(template_patch)
            
            output_sigma = np.sqrt(np.prod(target_size)) * self.cf_output_sigma_factor
            y_label = self._create_cf_gaussian_label(target_size, output_sigma)
            
            window = self._create_cf_hanning_window(target_size)
            features_windowed = features * window[:, :, np.newaxis]
            
            self.cf_model = self._train_cf(features_windowed, y_label)
            self.cf_template = features_windowed
            
        except Exception as e:
            print(f"相关滤波初始化失败: {e}")
            self.cf_enabled = False
    
    def apply_cf_refinement(self, image, predicted_bbox):
        """应用相关滤波进行精定位"""
        if not self.cf_enabled or self.cf_model is None or self.cf_template is None:
            return predicted_bbox
            
        try:
            x, y, w, h = [int(coord) for coord in predicted_bbox]
            
            search_factor = 1.5
            search_w = int(w * search_factor)
            search_h = int(h * search_factor)
            search_x = max(0, int(x + w/2 - search_w/2))
            search_y = max(0, int(y + h/2 - search_h/2))
            
            search_x = min(search_x, image.shape[1] - 1)
            search_y = min(search_y, image.shape[0] - 1)
            search_w = min(search_w, image.shape[1] - search_x)
            search_h = min(search_h, image.shape[0] - search_y)
            
            search_patch = image[search_y:search_y+search_h, search_x:search_x+search_w]
            
            if search_patch.size == 0:
                return predicted_bbox
                
            search_patch = cv2.resize(search_patch, (64, 64))
            features = self._extract_cf_features(search_patch)
            
            window = self._create_cf_hanning_window((64, 64))
            features_windowed = features * window[:, :, np.newaxis]
            
            best_response = -float('inf')
            best_position = (0, 0)
            
            for i in range(features_windowed.shape[2]):
                k = self._linear_correlation(features_windowed[:, :, i], self.cf_template[:, :, i])
                response = np.real(np.fft.ifft2(np.fft.fft2(k) * self.cf_model))
                
                max_idx = np.unravel_index(np.argmax(response), response.shape)
                max_response = response[max_idx]
                
                if max_response > best_response:
                    best_response = max_response
                    best_position = max_idx
            
            scale_x = search_w / 64.0
            scale_y = search_h / 64.0
            
            refined_center_x = search_x + best_position[1] * scale_x
            refined_center_y = search_y + best_position[0] * scale_y
            
            refined_bbox = [
                refined_center_x - w/2,
                refined_center_y - h/2,
                w,
                h
            ]
            
            return refined_bbox
            
        except Exception as e:
            print(f"相关滤波精定位失败: {e}")
            return predicted_bbox

    def add_template_to_library(self, template, sequence_id):
        """添加模板到模板库"""
        if sequence_id not in self.template_library:
            self.template_library[sequence_id] = []
        
        if len(self.template_library[sequence_id]) >= self.max_templates_per_sequence:
            self.template_library[sequence_id].pop(0)
        
        self.template_library[sequence_id].append(template.clone())
        print(f"模板库更新，当前模板数: {len(self.template_library[sequence_id])}")

    def global_redetection_with_template_library(self, image, current_state):
        """使用模板库进行全局重检测"""
        if self.current_sequence_id not in self.template_library or not self.template_library[self.current_sequence_id]:
            return current_state, 0.0
        
        best_state = current_state.copy()
        best_score = 0.0
        
        for i, template in enumerate(self.template_library[self.current_sequence_id]):
            try:
                expanded_search_factor = self.params.search_factor * 3.0
                x_patch_arr, resize_factor, x_amask_arr = sample_target(
                    image, current_state, expanded_search_factor,
                    output_sz=self.params.search_size
                )
                search = self.preprocessor.process(x_patch_arr)
                
                with torch.no_grad():
                    out_dict = self.network.forward_test(
                        template, search, self.text, self.prompt, self.flag
                    )
                
                pred_boxes = out_dict['bbox_map'].view(-1, 4).detach().cpu()
                pred_cls = out_dict['cls_score_test'].view(-1).detach().cpu()
                pred_cont = out_dict['cont_score'].softmax(-1)[:, :, 0].view(-1).detach().cpu() if self.has_cont else 1
                pred_cls_merge = pred_cls * self.window * pred_cont
                pred_box_net = pred_boxes[torch.argmax(pred_cls_merge)]
                nn_score = (pred_cls * pred_cont)[torch.argmax(pred_cls_merge)]
                
                pred_box = (pred_box_net * self.params.search_size / resize_factor).tolist()
                new_state = clip_box(self.map_box_back(pred_box, resize_factor), image.shape[0], image.shape[1], margin=10)
                
                if nn_score > best_score:
                    best_score = nn_score
                    best_state = new_state
                    print(f"模板 {i} 重检测得分: {nn_score:.3f}")
                    
            except Exception as e:
                print(f"模板 {i} 重检测失败: {e}")
                continue
        
        return best_state, best_score

    def _predict_with_flow(self, flow):
        """使用光流预测目标位置"""
        x, y, w, h = self.prev_bbox
        center_x = x + w/2
        center_y = y + h/2
        
        roi = flow[int(y):int(y+h), int(x):int(x+w)]
        if roi.size == 0:
            return self.prev_bbox 
        
        mean_flow = np.mean(roi, axis=(0, 1))
        
        new_center_x = center_x + mean_flow[0]
        new_center_y = center_y + mean_flow[1]
        
        new_x = new_center_x - w/2
        new_y = new_center_y - h/2
        
        return [new_x, new_y, w, h]
        
    def _draw_optical_flow(self, flow, image):
        """绘制彩色光流场"""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        overlay = cv2.addWeighted(image, 0.7, flow_rgb, 0.3, 0)
        
        return overlay, flow_rgb

    def _should_use_auxiliary_methods(self, nn_score):
        """判断是否需要使用辅助方法（光流+相关滤波）"""
        # 高置信度时完全信任网络
        if nn_score > self.update_threshold:
            return False
        # 极低置信度时触发重检测而不是辅助方法
        if nn_score < self.redetection_threshold:
            return False
        # 中等置信度时使用辅助方法
        return True
        
    def grounding(self, image, info: dict):
        bbox = torch.tensor([0., 0., 0., 0.]).cuda()
        h, w = image.shape[:2]
        im_crop_padded, _, _, _, _ = grounding_resize(image, self.params.grounding_size, bbox, None)
        ground = self.preprocessor.process(im_crop_padded).cuda()
        template = torch.zeros([1, 3, self.params.template_size, self.params.template_size]).cuda()
        template_mask = torch.zeros([1, (self.params.template_size//16)**2]).bool().cuda()
        context_mask = torch.zeros([1, (self.params.search_size//16)**2]).bool().cuda()
        
        # 修复：确保文本张量有正确的形状
        text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
        
        # 添加批次维度（如果不存在）
        if text.dim() == 1:
            text = text.unsqueeze(0)  # 从 [seq_len] 变为 [1, seq_len]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)  # 从 [seq_len] 变为 [1, seq_len]
        
        self.text = NestedTensor(text, mask)
        flag = torch.tensor([[1]]).cuda()
        
        with torch.no_grad():
            out_dict = self.network.forward(template, ground, self.text, template_mask, context_mask, flag)
        
        out_dict['pred_boxes'] = box_cxcywh_to_xywh(out_dict['pred_boxes']*np.max(image.shape[:2]))[0, 0].cpu().tolist()
        dx, dy = min(0, (w-h)/2), min(0, (h-w)/2)
        out_dict['pred_boxes'][0] = out_dict['pred_boxes'][0] + dx
        out_dict['pred_boxes'][1] = out_dict['pred_boxes'][1] + dy
        return out_dict

    def window_prior(self):
        hanning = np.hanning(self.map_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.torch_window = hann2d(torch.tensor([self.map_size, self.map_size]).long(), centered=True).flatten()

    def initialize(self, image, info: dict):
        if self.cfg.TEST.MODE == 'NL':
            grounding_state = self.grounding(image, info)
            init_bbox = grounding_state['pred_boxes']
            self.flag = torch.tensor([[2]]).cuda()
        elif self.cfg.TEST.MODE == 'NLBBOX':
            text, mask = self.extract_token_from_nlp(info['language'], self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN)
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[2]]).cuda()
        else:
            text = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).long().cuda()
            mask = torch.zeros([1, self.cfg.MODEL.BACKBONE.LANGUAGE.BERT.MAX_QUERY_LEN]).cuda()
            self.text = NestedTensor(text, mask)
            init_bbox = info['init_bbox']
            self.flag = torch.tensor([[0]]).cuda()
            
        self.window_prior()
        z_patch_arr, _, _, bbox = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size, return_bbox=True)
        self.template_mask = self.anno2mask(bbox.reshape(1, 4), size=self.params.template_size//16)
        self.z_patch_arr = z_patch_arr
        self.template_bbox = (bbox*self.params.template_size)[0, 0].tolist()

        self.template = self.preprocessor.process(z_patch_arr)
        self.dynamic_template = self.template.clone()

        y_patch_arr, _, _, y_bbox = sample_target(image, init_bbox, self.params.search_factor,
                                                    output_sz=self.params.search_size, return_bbox=True)
        self.y_patch_arr = y_patch_arr
        self.context_bbox = (y_bbox*self.params.search_size)[0, 0].tolist()
        context = self.preprocessor.process(y_patch_arr)
        context_mask = self.anno2mask(y_bbox.reshape(1, 4), self.params.search_size//16)
        self.prompt = self.network.forward_prompt_init(self.template, context, self.text, self.template_mask, context_mask, self.flag)
        
        self.state = init_bbox
        self.prev_bbox = init_bbox.copy()
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.cf_enabled:
            self.initialize_cf(image, self.state)

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow_prediction = None
        flow_visualization = None
        
        # 采样搜索区域图像块
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr)
        
        # 网络推理 - 核心定位
        with torch.no_grad():
            out_dict = self.network.forward_test(
                self.dynamic_template, search, self.text, self.prompt, self.flag
            )

        # 处理网络输出
        pred_boxes = out_dict['bbox_map'].view(-1, 4).detach().cpu()
        pred_cls = out_dict['cls_score_test'].view(-1).detach().cpu()
        pred_cont = out_dict['cont_score'].softmax(-1)[:, :, 0].view(-1).detach().cpu() if self.has_cont else 1

        pred_cls_merge = pred_cls * self.window * pred_cont
        pred_box_net = pred_boxes[torch.argmax(pred_cls_merge)]
        nn_score = (pred_cls * pred_cont)[torch.argmax(pred_cls_merge)]
        pred_box = (pred_box_net * self.params.search_size / resize_factor).tolist()
        
        # 裁剪到图像边界内
        tracker_prediction = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10
        )
        
        # ==================== 智能决策：何时使用辅助方法 ====================
        final_prediction = tracker_prediction
        use_auxiliary_methods = self._should_use_auxiliary_methods(nn_score)
        
        # 只在需要时计算光流和相关滤波
        if use_auxiliary_methods and self.use_optical_flow and self.prev_gray is not None and self.prev_bbox is not None:
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                **self.flow_params
            )
            flow_prediction = self._predict_with_flow(flow)
            
            # 加权融合光流预测和网络预测
            fused_prediction = [
                self.flow_smoothing_factor * flow_prediction[0] + (1 - self.flow_smoothing_factor) * tracker_prediction[0],
                self.flow_smoothing_factor * flow_prediction[1] + (1 - self.flow_smoothing_factor) * tracker_prediction[1],
                self.flow_smoothing_factor * flow_prediction[2] + (1 - self.flow_smoothing_factor) * tracker_prediction[2],
                self.flow_smoothing_factor * flow_prediction[3] + (1 - self.flow_smoothing_factor) * tracker_prediction[3]
            ]
            
            # 使用相关滤波对融合结果进行精定位
            cf_refined_prediction = self.apply_cf_refinement(image, fused_prediction)
            
            # 安全约束：防止修正幅度过大
            dx = cf_refined_prediction[0] - tracker_prediction[0]
            dy = cf_refined_prediction[1] - tracker_prediction[1]
            max_jump = min(W, H) * 0.1
            
            if dx**2 + dy**2 < max_jump**2:
                final_prediction = cf_refined_prediction
                print(f"Frame {self.frame_id}: 使用光流+相关滤波融合 (NN置信度: {nn_score:.3f})")
            #else:
                #print(f"Frame {self.frame_id}: 跳跃过大，使用网络预测")
        
        # ==================== 方位变化率估计和修正 ====================
        # 计算当前帧的方位向量（二维简化版）
        current_center = (final_prediction[0] + final_prediction[2]/2, 
                        final_prediction[1] + final_prediction[3]/2)
        image_center = (W/2, H/2)

        # 计算从图像中心指向目标中心的方位向量
        current_bearing = np.array([
            current_center[0] - image_center[0],
            current_center[1] - image_center[1]
        ])
        current_bearing = current_bearing / (np.linalg.norm(current_bearing) + 1e-8)

        if self.prev_bearing is not None and use_auxiliary_methods:
            print("sssss")
            # 计算方位变化率（导数近似）
            dt = 1.0  # 假设帧间时间间隔为1
            bearing_rate = (current_bearing - self.prev_bearing) / dt
            
            # 初始化或更新滤波器
            if self.bearing_rate_filter is None:
                self.bearing_rate_filter = bearing_rate.copy()
            else:
                # 简单的一阶低通滤波
                self.bearing_rate_filter = (self.bearing_rate_smoothing * bearing_rate + 
                                        (1 - self.bearing_rate_smoothing) * self.bearing_rate_filter)
            
            # 使用方位变化率信息修正预测结果（简化版STT-R思想）
            if np.linalg.norm(self.bearing_rate_filter) > 0.01:  # 只有变化明显时才修正
                # 计算速度向量（基于方位变化率的近似）
                estimated_velocity = self.bearing_rate_filter * 50.0  # 缩放因子，需要根据实际情况调整
                
                # 对预测结果进行小幅修正
                velocity_correction = estimated_velocity * 0.1  # 修正权重
                corrected_prediction = [
                    final_prediction[0] + velocity_correction[0],
                    final_prediction[1] + velocity_correction[1],
                    final_prediction[2],
                    final_prediction[3]
                ]
                
                # 确保修正后的边界框在图像范围内
                corrected_prediction = clip_box(corrected_prediction, H, W, margin=5)
                
                # 只在修正幅度合理时应用
                dx = corrected_prediction[0] - final_prediction[0]
                dy = corrected_prediction[1] - final_prediction[1]
                max_correction = min(W, H) * 0.05
                
                if dx**2 + dy**2 < max_correction**2:
                    final_prediction = corrected_prediction
                    print(f"Frame {self.frame_id}: 方位变化率修正 applied (rate: {np.linalg.norm(self.bearing_rate_filter):.4f})")
        
        # 更新上一帧的方位向量
        self.prev_bearing = current_bearing.copy()
        
        # 更新状态
        self.state = final_prediction
        
        # #模板更新决策
        # if nn_score > self.update_threshold:
        #     print(f"Frame {self.frame_id}: 模板更新, 神经网络置信度: {nn_score:.3f}")
        #     new_template_patch, _, _ = sample_target(image, final_prediction, self.params.template_factor,
        #                                             output_sz=self.params.template_size)
        #     self.dynamic_template = self.preprocessor.process(new_template_patch)
            
        #     if self.current_sequence_id is not None:
        #         self.add_template_to_library(self.dynamic_template, self.current_sequence_id)
            
        #     if self.cf_enabled:
        #         self.initialize_cf(image, final_prediction)
        
        # 重检测决策
        if nn_score < self.redetection_threshold:
            print(f"Frame {self.frame_id}: 触发重检测 (神经网络置信度: {nn_score:.3f})")
            original_state = final_prediction.copy()
            original_score = nn_score
            
            new_state, new_score = self.global_redetection_with_template_library(image, original_state)
            
            if new_score > original_score and new_score > 0.2:
                self.state = new_state
                final_prediction = new_state
                print(f"Frame {self.frame_id}: 重检测成功, 新得分: {new_score:.3f}")
                
                if new_score > self.update_threshold * 0.8:
                    new_template_patch, _, _ = sample_target(
                        image, final_prediction, self.params.template_factor,
                        output_sz=self.params.template_size
                    )
                    self.dynamic_template = self.preprocessor.process(new_template_patch)
                    if self.cf_enabled:
                        self.initialize_cf(image, final_prediction)
            #else:
                #print(f"Frame {self.frame_id}: 重检测未找到更优结果")
        
        # 更新前一帧数据（供下一帧使用）
        self.prev_bbox = final_prediction
        self.prev_gray = gray
        
        # 可视化
        vis_image = image.copy()
        
        # 绘制网络预测框
        cv2.rectangle(vis_image, 
                    (int(tracker_prediction[0]), int(tracker_prediction[1])),
                    (int(tracker_prediction[0] + tracker_prediction[2]), 
                    int(tracker_prediction[1] + tracker_prediction[3])),
                    self.colors['tracker'], 2)
        cv2.putText(vis_image, f"NN: {nn_score:.2f}", 
                (int(tracker_prediction[0]), int(tracker_prediction[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['tracker'], 1)
        
        # 只有在使用了辅助方法时才绘制相关框
        if use_auxiliary_methods and flow_prediction is not None:
            cv2.rectangle(vis_image, 
                        (int(flow_prediction[0]), int(flow_prediction[1])),
                        (int(flow_prediction[0] + flow_prediction[2]), 
                        int(flow_prediction[1] + flow_prediction[3])),
                        self.colors['flow'], 2)
            cv2.putText(vis_image, "Flow", 
                    (int(flow_prediction[0]), int(flow_prediction[1])-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['flow'], 1)
            
            if final_prediction != tracker_prediction:
                cv2.rectangle(vis_image, 
                            (int(final_prediction[0]), int(final_prediction[1])),
                            (int(final_prediction[0] + final_prediction[2]), 
                            int(final_prediction[1] + final_prediction[3])),
                            self.colors['cf'], 2)
                cv2.putText(vis_image, "CF Refined", 
                        (int(final_prediction[0]), int(final_prediction[1])-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['cf'], 1)
        
        # 绘制最终预测框
        cv2.rectangle(vis_image, 
                    (int(final_prediction[0]), int(final_prediction[1])),
                    (int(final_prediction[0] + final_prediction[2]), 
                    int(final_prediction[1] + final_prediction[3])),
                    self.colors['fused'], 3)
        cv2.putText(vis_image, f"Final", 
                (int(final_prediction[0]), int(final_prediction[1])-70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['fused'], 2)
        
        # 如果有真实框，绘制真实框
        if info is not None and 'target_bbox' in info:
            gt_bbox = info['target_bbox']
            x, y, w, h = gt_bbox
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x+w), int(y+h)), 
                        self.colors['gt'], 2)
            cv2.putText(vis_image, "GT", 
                    (int(x), int(y)-90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['gt'], 2)
        
        # 轨迹绘制
        current_center = (int(final_prediction[0] + final_prediction[2]/2), 
                        int(final_prediction[1] + final_prediction[3]/2))
        self.trajectory.append(current_center)
        if len(self.trajectory) > self.trajectory_length:
            self.trajectory.pop(0)
        
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                cv2.line(vis_image, self.trajectory[i-1], self.trajectory[i], 
                        self.colors['trajectory'], self.trajectory_thickness)
        
        cv2.circle(vis_image, current_center, 5, self.colors['center'], -1)
        
        # 添加信息文本
        cv2.putText(vis_image, f"Frame: {self.frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Score: {nn_score:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示方位变化率信息（如果计算了）
        if self.prev_bearing is not None and use_auxiliary_methods and self.bearing_rate_filter is not None:
            cv2.putText(vis_image, f"Bearing Rate: {np.linalg.norm(self.bearing_rate_filter):.3f}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 绘制方位变化率向量
            center_point = (int(image_center[0]), int(image_center[1]))
            rate_vector = (int(center_point[0] + self.bearing_rate_filter[0] * 100),
                        int(center_point[1] + self.bearing_rate_filter[1] * 100))
            cv2.arrowedLine(vis_image, center_point, rate_vector, 
                        (0, 255, 255), 2, tipLength=0.3)
        
        # 保存可视化结果
        if self.debug:
            cv2.imwrite(f"{self.sequence_save_dir}/frame_{self.frame_id}.jpg", vis_image)
        
        # 返回结果
        result = {"target_bbox": self.state, "best_score": nn_score, "visualization": vis_image}
        if flow_visualization is not None:
            result["flow_visualization"] = flow_visualization
        
        return result
    
    
    def calculate_iou(self, box1, box2):
        """计算两个边界框之间的IOU值"""
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def extract_token_from_nlp(self, nlp, seq_length):
        nlp_token = self.tokenizer(nlp, return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)
        text = nlp_token['input_ids'].squeeze(0).cuda()
        mask = nlp_token['attention_mask'].squeeze(0).cuda()
        return text, mask

    def anno2mask(self, bbox, size):
        mask = torch.zeros([1, size, size])
        bbox = bbox * size
        bbox = bbox.int()
        mask[:, bbox[0, 1]:bbox[0, 3], bbox[0, 0]:bbox[0, 2]] = 1
        mask = mask.view(1, -1).bool().cuda()
        return mask

def get_tracker_class():
    return METrack