import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gc
import math
from collections import deque
from ultralytics import YOLO
import mediapipe as mp


# ================= 1. LSTM 模型定义 (保持原样) =================
class OriginalTennisLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=132, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, x, lengths): return self.classifier(x[:, -1, :])


class TennisKnowledgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rule_net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, keypoints): return self.rule_net(torch.zeros(keypoints.shape[0], 6).to(keypoints.device))


class KnowledgeFusedLSTM(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.knowledge_encoder = TennisKnowledgeEncoder()
        self.classifier = nn.Sequential(nn.Linear(128 + 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32),
                                        nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x, lengths):
        batch_size, max_len, _, _ = x.shape
        x_flat = x.reshape(batch_size, max_len, -1)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x_sorted = x_flat[sorted_idx]
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths.cpu(), batch_first=True,
                                                           enforce_sorted=True)
        packed_out, _ = self.original_model.lstm(packed_x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)
        _, original_idx = torch.sort(sorted_idx)
        out = out[original_idx]
        attention_scores = self.original_model.attention(out)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        weighted = (out * attention_weights).sum(dim=1)
        k_features = self.knowledge_encoder(x)
        l_features = self.original_model.classifier[0](weighted)
        l_features = self.original_model.classifier[1](l_features)
        l_features = self.original_model.classifier[2](l_features)
        return self.classifier(torch.cat([l_features, k_features], dim=1))


# ================= 2. 辅助函数 =================
def get_center(bbox): return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def get_area(bbox): return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


# ================= 3. TennisSystem 主类 =================
class TennisSystem:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 路径配置
        self.PLAYER_MODEL = os.path.join(model_dir, "yolov8x.pt")
        self.BALL_MODEL = os.path.join(model_dir, "yolov8_last.pt")
        self.ORIGINAL_MODEL_PATH = os.path.join(model_dir, "best_model.pth")
        self.FUSED_MODEL_PATH = os.path.join(model_dir, "knowledge_fused_best.pth")

        # 参数
        self.PLAYER_CONF = 0.3
        self.BALL_CONF = 0.20
        self.CLASS_NAMES = ['Forehand', 'Backhand', 'Serve']
        self.CLASS_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

        print(f"[系统] 初始化 TennisSystem (Device: {self.device})...")

        try:
            self.p_model = YOLO(self.PLAYER_MODEL)
            self.b_model = YOLO(self.BALL_MODEL)
            print("  ✓ YOLO 模型加载成功")
        except Exception as e:
            print(f"  ❌ YOLO 模型加载失败: {e}")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1)
        self.classifier_model = self._load_classifier()
        self.ball_trail = deque(maxlen=8)

        # 可视化开关
        self.enable_skeleton = True      # 在视频上叠加骨架
        self.enable_mini_court = True    # 在右上角显示小球场

        # 估计出来的「像素 -> 米」尺度（只用于显示球速）
        self.px_to_m = None

    def _load_classifier(self):
        if not os.path.exists(self.FUSED_MODEL_PATH):
            return None
        try:
            orig = OriginalTennisLSTM()
            if os.path.exists(self.ORIGINAL_MODEL_PATH):
                ckpt = torch.load(self.ORIGINAL_MODEL_PATH, map_location='cpu')
                orig.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

            model = KnowledgeFusedLSTM(orig)
            ckpt_f = torch.load(self.FUSED_MODEL_PATH, map_location='cpu')
            model.load_state_dict(ckpt_f['model_state_dict'] if 'model_state_dict' in ckpt_f else ckpt_f)
            model.to(self.device).eval()
            print("  ✓ LSTM 分类模型加载成功")
            return model
        except Exception as e:
            print(f"  ❌ LSTM 加载出错: {e}")
            return None

    def process_video(self, input_path, output_path):
        print(f"⚡ 开始处理视频: {input_path}")
        cap = cv2.VideoCapture(input_path)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        # 1. 检测
        player_boxes, ball_centers = self._run_yolo(frames)
        ball_smooth = self._interpolate_ball(ball_centers)

        # 2. 物理逻辑 (使用完整版)
        segments = self._detect_physics_logic(ball_smooth, player_boxes, len(frames))

        # 3. 分类 + 骨骼提取
        shots_data = []
        if segments and self.classifier_model:
            shots_data = self._classify_and_extract(input_path, segments)

        # 4. 渲染
        self._render(input_path, output_path, shots_data, player_boxes, ball_centers)

        del frames
        gc.collect()

        return {"output_video": output_path, "shots": shots_data}

    def _run_yolo(self, frames):
        p_boxes, b_centers = [], []
        h, w = frames[0].shape[:2]
        area_thresh = (h * w) * 0.008

        for i, frame in enumerate(frames):
            p_res = self.p_model(frame, verbose=False, conf=self.PLAYER_CONF)[0]
            best_p = None
            max_area = 0
            if p_res.boxes:
                for box in p_res.boxes:
                    if int(box.cls[0]) == 0:
                        xy = box.xyxy[0].tolist()
                        area = get_area(xy)
                        if area > area_thresh and area > max_area:
                            max_area = area
                            best_p = xy
            p_boxes.append([best_p] if best_p else [])

            b_res = self.b_model(frame, verbose=False, conf=self.BALL_CONF)[0]
            best_c = -1
            center = None
            if b_res.boxes:
                for box in b_res.boxes:
                    if box.conf[0] > best_c:
                        best_c = float(box.conf[0])
                        xy = box.xyxy[0].tolist()
                        center = get_center(xy)
            b_centers.append(center)
            if i % 50 == 0:
                print(f"\r  YOLO: {i}/{len(frames)}", end="")
        print("")
        return p_boxes, b_centers

    def _interpolate_ball(self, centers):
        clean = [[c[0], c[1]] if c else [np.nan, np.nan] for c in centers]
        df = pd.DataFrame(clean, columns=['x', 'y'])
        df = df.interpolate(method='linear', limit=6, limit_direction='both')
        df['x'] = df['x'].rolling(window=3, center=True, min_periods=1).mean()
        df['y'] = df['y'].rolling(window=3, center=True, min_periods=1).mean()
        res = []
        for _, r in df.iterrows():
            if np.isnan(r['x']):
                res.append(None)
            else:
                res.append((int(r['x']), int(r['y'])))
        return res

    def _detect_physics_logic(self, ball_centers, player_boxes, total_frames):
        """
        完整版物理检测逻辑
        包含：击球检测、地面过滤、片段扩展、合并重叠
        """
        # 参数配置 (与你的 main.py 保持一致)
        DIST_THRESH = 240
        COS_SIM_THRESH = -0.25
        ACCEL_RATIO_THRESH = 1.5
        MIN_SPEED_THRESH = 3.0
        MIN_HIT_GAP = 8
        GROUND_FILTER_RATIO = 0.15
        ACTION_DISTANCE_THRESH = 320
        EXTRA_PADDING_PRE = 8
        EXTRA_PADDING_POST = 15

        hits = []
        last_frame = -100
        step = 2

        # 1. 击球候选检测循环
        for i in range(step, total_frames - step):
            curr = ball_centers[i]
            prev = ball_centers[i - step]
            next_p = ball_centers[i + step]

            # 基础有效性检查
            if not (curr and prev and next_p and player_boxes[i]):
                continue

            p_box = player_boxes[i][0]
            p_cx, p_cy = get_center(p_box)

            # 距离检查
            if np.linalg.norm(np.array(curr) - np.array([p_cx, p_cy])) > DIST_THRESH:
                continue

            # 地面过滤 (防止误检反弹球)
            if curr[1] > (p_box[3] - (p_box[3] - p_box[1]) * GROUND_FILTER_RATIO):
                continue

            # 物理向量计算
            vec_in = np.array(curr) - np.array(prev)
            vec_out = np.array(next_p) - np.array(curr)
            speed_in = np.linalg.norm(vec_in)
            speed_out = np.linalg.norm(vec_out)

            if speed_out < MIN_SPEED_THRESH:
                continue

            denom = (speed_in * speed_out) + 1e-6
            cos_sim = np.dot(vec_in, vec_out) / denom

            is_angle = (cos_sim < COS_SIM_THRESH)
            is_accel = (speed_out / (speed_in + 1e-6) > ACCEL_RATIO_THRESH) and (speed_out > 8.0)

            if (is_angle or is_accel) and (i - last_frame > MIN_HIT_GAP):
                hits.append({'frame': i, 'conf': 0.85 if is_accel else abs(cos_sim)})
                last_frame = i

        # 2. 生成片段 (向前向后搜索)
        segments = []
        for h in hits:
            f = h['frame']
            if not player_boxes[f]:
                continue
            p_c = get_center(player_boxes[f][0])

            # 向前搜索开始帧
            start = max(0, f - 10)
            for k in range(f, max(0, f - 40), -1):
                if ball_centers[k] and np.linalg.norm(
                        np.array(ball_centers[k]) - np.array(p_c)) > ACTION_DISTANCE_THRESH:
                    start = k
                    break

            # 向后搜索结束帧
            end = min(total_frames - 1, f + 15)
            for k in range(f, min(total_frames, f + 40)):
                if ball_centers[k] and np.linalg.norm(
                        np.array(ball_centers[k]) - np.array(p_c)) > ACTION_DISTANCE_THRESH:
                    end = k
                    break

            s_final = max(0, start - EXTRA_PADDING_PRE)
            e_final = min(total_frames - 1, end + EXTRA_PADDING_POST)

            segments.append({
                'contact_frame': f,
                'start_frame': s_final,
                'end_frame': e_final,
                'segment_frames': list(range(s_final, e_final + 1)),
                'sequence_length': e_final - s_final + 1,
                'confidence': h['conf']
            })

        # 3. 合并重叠片段
        if not segments:
            return []
        segments.sort(key=lambda x: x['start_frame'])
        merged = [segments[0]]
        for curr in segments[1:]:
            last = merged[-1]
            if curr['start_frame'] <= last['end_frame'] + 5:
                # 合并逻辑
                last['end_frame'] = max(last['end_frame'], curr['end_frame'])
                last['segment_frames'] = list(range(last['start_frame'], last['end_frame'] + 1))
                last['sequence_length'] = len(last['segment_frames'])
                # 保留置信度更高的击球点
                if curr['confidence'] > last['confidence']:
                    last['contact_frame'] = curr['contact_frame']
                    last['confidence'] = curr['confidence']
            else:
                merged.append(curr)

        return merged

    def _classify_and_extract(self, video_path, segments):
        cap = cv2.VideoCapture(video_path)
        results = []
        print(f"  分类分析: {len(segments)} 个动作")

        for idx, seg in enumerate(segments):
            kps_seq = []
            for f_idx in seg['segment_frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = self.pose.process(rgb)
                    if res.pose_landmarks:
                        kps_seq.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark])
                    else:
                        kps_seq.append(np.zeros((33, 4)).tolist())
                else:
                    kps_seq.append(np.zeros((33, 4)).tolist())

            inp = torch.FloatTensor(np.array(kps_seq)).unsqueeze(0).to(self.device)
            slen = torch.tensor([len(kps_seq)]).to(self.device)
            with torch.no_grad():
                out = self.classifier_model(inp, slen)
                probs = torch.nn.functional.softmax(out, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            results.append({
                'start_frame': seg['start_frame'],
                'hit_frame': seg['contact_frame'],
                'end_frame': seg['end_frame'],
                'type': self.CLASS_NAMES[pred],
                'confidence': float(probs[0, pred].item()),
                'kps_seq': kps_seq,
                '_seq_id': idx + 1  # 给 HUD 用的片段编号
            })
        cap.release()
        return results

    # ============== 骨架绘制函数 ==============
    def _draw_pose_skeleton(self, frame, kps_seq_frame):
        """
        使用 MediaPipe Pose 的关键点在画面上画出骨架轮廓.
        kps_seq_frame: 单帧 33x4 列表 [x, y, z, visibility]，坐标是 0~1 归一化.
        """
        if kps_seq_frame is None:
            return

        h, w = frame.shape[:2]
        points = []
        for lm in kps_seq_frame:
            x, y, z, v = lm
            # 置信度太低的点忽略
            if v < 0.3:
                points.append(None)
                continue
            px, py = int(x * w), int(y * h)
            if 0 <= px < w and 0 <= py < h:
                points.append((px, py))
            else:
                points.append(None)

        # 先画骨架连线
        for conn in self.mp_pose.POSE_CONNECTIONS:
            s = getattr(conn[0], "value", int(conn[0]))
            e = getattr(conn[1], "value", int(conn[1]))
            if s < len(points) and e < len(points):
                p1, p2 = points[s], points[e]
                if p1 is not None and p2 is not None:
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # 再画每个关键点
        for pt in points:
            if pt is not None:
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)

    # ============== 球场 ROI 估计（用于小球场缩略图 & 尺度估计） ==============
    def _estimate_court_roi(self, frame):
        """
        根据首帧白线自动估计球场区域 (x0, y0, x1, y1)。

        只用于可视化缩略小球场和球速，大逻辑不依赖这个结果。
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 亮度高、饱和度低的“白线”
        lower = np.array([0, 0, 180], dtype=np.uint8)
        upper = np.array([179, 80, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 只关注画面中下部区域，排除看台
        y_top = int(h * 0.25)
        y_bottom = int(h * 0.98)
        x_left = int(w * 0.05)
        x_right = int(w * 0.95)

        focus = np.zeros_like(mask)
        focus[y_top:y_bottom, x_left:x_right] = mask[y_top:y_bottom, x_left:x_right]

        kernel = np.ones((5, 5), np.uint8)
        focus = cv2.morphologyEx(focus, cv2.MORPH_CLOSE, kernel, iterations=2)

        ys, xs = np.where(focus > 0)
        if len(xs) == 0 or len(ys) == 0:
            # 兜底：没有检测到白线，就用中间大致区域
            return x_left, y_top, x_right, y_bottom

        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        margin_x = int((x_max - x_min) * 0.05)
        margin_y = int((y_max - y_min) * 0.05)

        x0 = max(x_left + x_min - margin_x, 0)
        x1 = min(x_left + x_max + margin_x, w - 1)
        y0 = max(y_top + y_min - margin_y, 0)
        y1 = min(y_top + y_max + margin_y, h - 1)

        return int(x0), int(y0), int(x1), int(y1)

    # ============== 右上角小球场（基于 ROI，纯线条版） ==============
    def _draw_mini_court(self, frame, court_roi, player_center, ball_center):
        """
        在右上角绘制小球场缩略图（纯线条版本）。
        court_roi: 原画面中估计出的球场 ROI，用于把球/球员位置投影到小球场上。
        """
        if court_roi is None:
            return

        h, w = frame.shape[:2]
        x0, y0, x1, y1 = court_roi

        # 边界保护
        x0 = max(0, min(int(x0), w - 2))
        x1 = max(1, min(int(x1), w - 1))
        y0 = max(0, min(int(y0), h - 2))
        y1 = max(1, min(int(y1), h - 1))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return

        court_w = x1 - x0
        court_h = y1 - y0
        if court_w <= 0 or court_h <= 0:
            return

        # 小球场尺寸：宽度固定为画面的约 22%，高度按球场纵横比自适应
        mini_w = int(w * 0.22)
        mini_h = int(mini_w * court_h / float(court_w))
        mini_h = max(50, min(mini_h, int(h * 0.35)))

        x_start = w - mini_w - 20
        y_start = 20
        x_end = x_start + mini_w
        y_end = y_start + mini_h

        # 外层黑色蒙版
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 6, y_start - 6),
                      (x_end + 6, y_end + 6), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # -------- 纯线条小球场背景 --------
        # 统一颜色背景 + 网线、边线等
        mini = np.zeros((mini_h, mini_w, 3), dtype=np.uint8)
        # 网球场底色（略偏蓝），如果你想要完全黑底把下面一行改成 (0, 0, 0)
        mini[:] = (40, 80, 160)

        white = (255, 255, 255)
        thickness = 1 if min(mini_w, mini_h) < 120 else 2

        margin_x = int(mini_w * 0.08)
        margin_y = int(mini_h * 0.08)
        left = margin_x
        right = mini_w - margin_x
        top = margin_y
        bottom = mini_h - margin_y

        # 外边线（双打边线）
        cv2.rectangle(mini, (left, top), (right, bottom), white, thickness)

        court_len = bottom - top
        court_wid = right - left

        # 中网
        net_y = int(top + court_len * 0.5)
        cv2.line(mini, (left, net_y), (right, net_y), white, thickness)

        # 单打边线（内缩一点）
        singles_margin = int(court_wid * 0.09)
        s_left = left + singles_margin
        s_right = right - singles_margin
        cv2.line(mini, (s_left, top), (s_left, bottom), white, thickness)
        cv2.line(mini, (s_right, top), (s_right, bottom), white, thickness)

        # 发球线（两侧对称）
        service_offset = int(court_len * 0.21)  # 约等于真实比例
        service_y_top = net_y - service_offset
        service_y_bottom = net_y + service_offset
        cv2.line(mini, (s_left, service_y_top), (s_right, service_y_top), white, thickness)
        cv2.line(mini, (s_left, service_y_bottom), (s_right, service_y_bottom), white, thickness)

        # 中心发球线
        center_x = int((s_left + s_right) * 0.5)
        cv2.line(mini, (center_x, service_y_top), (center_x, service_y_bottom), white, thickness)

        # 底线中点的小标记
        mark_len = int(court_len * 0.03)
        cv2.line(mini, (center_x, top), (center_x, top + mark_len), white, thickness)
        cv2.line(mini, (center_x, bottom - mark_len), (center_x, bottom), white, thickness)

        # 把小球场贴回画面
        frame[y_start:y_end, x_start:x_end] = mini
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)

        # -------- 把真实球/球员位置投影到小球场上 --------
        def project(pt):
            if pt is None:
                return None
            px, py = pt
            if px < x0 or px > x1 or py < y0 or py > y1:
                return None
            nx = (px - x0) / float(x1 - x0)
            ny = (py - y0) / float(y1 - y0)
            mx = int(x_start + nx * mini_w)
            my = int(y_start + ny * mini_h)
            return mx, my

        p_pt = project(player_center)
        b_pt = project(ball_center)

        if p_pt is not None:
            cv2.circle(frame, p_pt, 4, (0, 255, 0), -1)
        if b_pt is not None:
            cv2.circle(frame, b_pt, 3, (0, 255, 255), -1)

    # ============== 专业 HUD 绘制 ==============
    def _draw_hud(self, frame, idx, total_frames, zone_map, type_map, ball_speed, speed_unit):
        """
        专业级 HUD：
        - 左上角：系统状态牌
        - 底部：击球类型 + 置信度 + 片段编号 + 时间轴进度 + 球速
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # -------- 左上角系统状态牌 --------
        panel_w, panel_h = 260, 70
        x0, y0 = 20, 20
        x1, y1 = x0 + panel_w, y0 + panel_h
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "AI Coach Pro", (x0 + 12, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Device: {self.device}", (x0 + 12, y0 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {idx+1}/{max(total_frames,1)}", (x0 + 12, y0 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # -------- 底部 HUD 条 --------
        overlay = frame.copy()
        bar_h = 80
        bx0, by0 = 20, h - bar_h - 10
        bx1, by1 = w - 20, h - 10
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # 左侧：当前动作信息 + 球速
        info = type_map.get(idx)
        if info:
            shot_type = info.get("type", "Unknown")
            conf = info.get("confidence", 0.0)
            seq_id = info.get("_seq_id", 1)

            main_color = self.CLASS_COLORS[self.CLASS_NAMES.index(shot_type)] if shot_type in self.CLASS_NAMES else (0, 255, 255)

            text_main = f"{shot_type}  #{seq_id}"
            cv2.putText(frame, text_main, (bx0 + 20, by0 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, main_color, 2, cv2.LINE_AA)

            text_sub = f"Confidence: {int(conf * 100)}%"
            cv2.putText(frame, text_sub, (bx0 + 22, by0 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No active shot", (bx0 + 20, by0 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA)

        # 球速显示（左侧下方）
        if ball_speed is not None:
            speed_text = f"Ball speed: {ball_speed:.1f} {speed_unit}"
            cv2.putText(frame, speed_text, (bx0 + 20, by0 + 69),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)

        # 中间：状态描述
        status_text = "Inside shot segment" if idx in zone_map else "Idle / Non-hit frame"
        status_color = (0, 255, 255) if idx in zone_map else (150, 150, 150)
        cv2.putText(frame, status_text, (w // 2 - 140, by0 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        # 右侧：时间轴进度条
        prog_x0 = w - 260
        prog_x1 = w - 40
        prog_y = by0 + 50
        cv2.line(frame, (prog_x0, prog_y), (prog_x1, prog_y), (80, 80, 80), 8)

        if total_frames > 1:
            ratio = idx / (total_frames - 1)
        else:
            ratio = 0.0
        cur_x = int(prog_x0 + ratio * (prog_x1 - prog_x0))

        cv2.line(frame, (prog_x0, prog_y), (cur_x, prog_y),
                 (0, 200, 255) if idx in zone_map else (120, 120, 120), 8)

        if info and idx == info.get("hit_frame"):
            cv2.circle(frame, (cur_x, prog_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cur_x, prog_y), 16, (0, 0, 255), 2)

    # ============== 渲染函数（骨架 + 小球场 + HUD + 球速） ==============
    def _render(self, input_path, output_path, results, p_boxes, b_centers):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(b_centers)
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1

        # 优先使用 avc1 (H.264)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # 预估球场 ROI（用于小球场 & 像素尺度）
        court_roi = None
        self.px_to_m = None
        ret0, first_frame = cap.read()
        if ret0:
            first_frame = first_frame[:h, :w]
            court_roi = self._estimate_court_roi(first_frame)
            # 用球场纵向长度估计像素到米：大致对应单打/双打底线到底线 23.77m
            if court_roi is not None:
                _, y0, _, y1 = court_roi
                pix_len = max(1, y1 - y0)
                self.px_to_m = 23.77 / float(pix_len)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        idx = 0
        zone_map = {}
        type_map = {}
        pose_map = {}

        for r in results:
            for f in range(r['start_frame'], r['end_frame'] + 1):
                zone_map[f] = True
                type_map[f] = r

            kps_seq = r.get('kps_seq')
            if kps_seq:
                for offset, kps in enumerate(kps_seq):
                    frame_idx = r['start_frame'] + offset
                    if frame_idx not in pose_map:
                        pose_map[frame_idx] = kps

        prev_ball = None
        smooth_speed = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:h, :w]

            # 当前帧球的位置
            curr = b_centers[idx] if idx < len(b_centers) else None

            # 计算球速（像素/帧 -> m/s -> km/h），简单指数平滑
            display_speed = None
            speed_unit = "km/h"
            if curr is not None and prev_ball is not None:
                dist_pix = float(np.linalg.norm(np.array(curr) - np.array(prev_ball)))
                if self.px_to_m is not None:
                    speed_mps = dist_pix * self.px_to_m * fps
                    inst_speed = speed_mps * 3.6  # km/h
                else:
                    inst_speed = dist_pix * fps   # 像素 / 秒
                    speed_unit = "px/s"
                smooth_speed = 0.7 * smooth_speed + 0.3 * inst_speed
                display_speed = smooth_speed
            elif smooth_speed > 0:
                # 球暂时没检测到时，用上一个速度慢慢衰减
                smooth_speed *= 0.95
                display_speed = smooth_speed
                speed_unit = "km/h" if self.px_to_m is not None else "px/s"

            prev_ball = curr if curr is not None else prev_ball

            # 画球 + 轨迹
            if curr:
                self.ball_trail.append(curr)
                cv2.circle(frame, curr, 5, (0, 255, 255), -1)
                # 在球旁边写个小的速度数字
                if display_speed is not None:
                    text = f"{display_speed:.0f}"
                    cv2.putText(frame, text, (curr[0] + 8, curr[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                self.ball_trail.append(None)
            for t in range(1, len(self.ball_trail)):
                if self.ball_trail[t - 1] and self.ball_trail[t]:
                    cv2.line(frame, self.ball_trail[t - 1],
                             self.ball_trail[t], (0, 200, 255), 2)

            # 当前帧球员 bbox + 中心
            player_center = None
            if idx < len(p_boxes) and p_boxes[idx]:
                bx = p_boxes[idx][0]
                cv2.rectangle(frame, (int(bx[0]), int(bx[1])),
                              (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
                player_center = get_center(bx)

            # 骨架
            if self.enable_skeleton and idx in pose_map:
                self._draw_pose_skeleton(frame, pose_map[idx])

            # 小球场缩略图（基于真实球场 ROI）
            if self.enable_mini_court:
                self._draw_mini_court(frame, court_roi, player_center, curr)

            # 击球片段高亮边框
            if idx in zone_map:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 4)

            # 中央击球类型 + HIT 提示
            if idx in type_map:
                info = type_map[idx]
                text = info['type']
                color = self.CLASS_COLORS[self.CLASS_NAMES.index(text)]

                (wt, ht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
                cx, cy = w // 2, h // 2
                overlay_center = frame.copy()
                cv2.rectangle(overlay_center, (cx - wt // 2 - 18, cy - ht - 14),
                              (cx + wt // 2 + 18, cy + 14), (0, 0, 0), -1)
                cv2.addWeighted(overlay_center, 0.55, frame, 0.45, 0, frame)

                cv2.putText(frame, text, (cx - wt // 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

                if idx == info['hit_frame']:
                    cv2.putText(frame, "HIT", (cx - 34, cy + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            # HUD（左上状态 + 底部信息条 + 球速）
            self._draw_hud(frame, idx, total_frames, zone_map, type_map,
                           display_speed, "km/h" if self.px_to_m is not None else "px/s")

            out.write(frame)
            idx += 1
            if idx % 50 == 0:
                print(f"\r  渲染: {idx}", end="")

        cap.release()
        out.release()
        print("\n  渲染完成")
