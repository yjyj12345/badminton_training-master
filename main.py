"""
Final Tennis Analysis System (Direct MP4 Edition)
Features:
1. Direct .mp4 generation using 'mp4v' or 'avc1'.
2. Force even-dimensions to prevent encoder crashes (CRITICAL FIX).
3. No MJPG, smaller file size.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import math
import pandas as pd
import gc
from collections import deque
from ultralytics import YOLO

# ===================== 1. Configuration =====================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- INPUT PATH ---
    INPUT_VIDEO = "/root/tennis_correct/data/videos_train/386.mp4"

    # --- OUTPUT PATHS ---
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")

    # [修改] 直接输出 MP4
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "2.mp4")
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, "2.json")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")

    # --- MODEL PATHS ---
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    if not os.path.exists(MODEL_DIR): MODEL_DIR = BASE_DIR

    PLAYER_MODEL = "yolov8x.pt"
    BALL_MODEL = "yolov8_last.pt"
    ORIGINAL_MODEL = "best_model.pth"
    FUSED_MODEL = "knowledge_fused_best.pth"

    # --- PARAMETERS ---
    PLAYER_CONF = 0.3
    BALL_CONF = 0.20
    DIST_THRESH = 240
    COS_SIM_THRESH = -0.25
    ACCEL_RATIO_THRESH = 1.5
    MIN_SPEED_THRESH = 3.0
    MIN_HIT_GAP = 8
    GROUND_FILTER_RATIO = 0.15
    ACTION_DISTANCE_THRESH = 320
    EXTRA_PADDING_PRE = 8
    EXTRA_PADDING_POST = 15

    CLASS_NAMES = ['Forehand', 'Backhand', 'Serve']
    CLASS_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== 2. Embedded Models =====================
class OriginalTennisLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=132, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
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
        self.classifier = nn.Sequential(nn.Linear(128 + 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x, lengths):
        batch_size, max_len, _, _ = x.shape
        x_flat = x.reshape(batch_size, max_len, -1)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        x_sorted = x_flat[sorted_idx]
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True)
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

# ===================== 3. Helper Functions =====================
def get_center(bbox): return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
def get_area(bbox): return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
def ensure_dir(d):
    if not os.path.exists(d): os.makedirs(d)

# ===================== 4. Detection System =====================
class AdvancedShotDetector:
    def __init__(self):
        print("\n[1/5] Initializing Detector...")
        self.p_model = YOLO(Config.PLAYER_MODEL)
        self.b_model = YOLO(Config.BALL_MODEL)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()

        print(f"  - Detecting on {len(frames)} frames...")
        player_boxes, ball_centers = self._run_yolo(frames)
        print("  - Smoothing Trajectory...")
        ball_smooth = self._interpolate_ball(ball_centers)
        print("  - Analyzing Physics...")
        segments = self._detect_logic(ball_smooth, player_boxes, len(frames))
        return segments, frames, fps, player_boxes, ball_smooth

    def _run_yolo(self, frames):
        player_boxes = []
        ball_centers = []
        h, w = frames[0].shape[:2]
        area_thresh = (h * w) * 0.008

        for i, frame in enumerate(frames):
            p_res = self.p_model(frame, verbose=False, conf=Config.PLAYER_CONF)[0]
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
            player_boxes.append([best_p] if best_p else [])
            b_res = self.b_model(frame, verbose=False, conf=Config.BALL_CONF)[0]
            best_c = -1
            center = None
            if b_res.boxes:
                for box in b_res.boxes:
                    if box.conf[0] > best_c:
                        best_c = float(box.conf[0])
                        xy = box.xyxy[0].tolist()
                        center = get_center(xy)
            ball_centers.append(center)
            if i % 100 == 0: print(f"\r    YOLO: {i}/{len(frames)}", end="")
        print("")
        return player_boxes, ball_centers

    def _interpolate_ball(self, centers):
        clean = [[c[0], c[1]] if c else [np.nan, np.nan] for c in centers]
        df = pd.DataFrame(clean, columns=['x', 'y'])
        df = df.interpolate(method='linear', limit=6, limit_direction='both')
        df['x'] = df['x'].rolling(window=3, center=True, min_periods=1).mean()
        df['y'] = df['y'].rolling(window=3, center=True, min_periods=1).mean()
        res = []
        for _, r in df.iterrows():
            if np.isnan(r['x']): res.append(None)
            else: res.append((int(r['x']), int(r['y'])))
        return res

    def _detect_logic(self, ball_centers, player_boxes, total_frames):
        hits = []
        last_frame = -100
        step = 2
        for i in range(step, total_frames - step):
            curr = ball_centers[i]
            prev = ball_centers[i-step]
            next_p = ball_centers[i+step]
            if not (curr and prev and next_p and player_boxes[i]): continue
            p_box = player_boxes[i][0]
            p_cx, p_cy = get_center(p_box)
            if np.linalg.norm(np.array(curr) - np.array([p_cx, p_cy])) > Config.DIST_THRESH: continue
            if curr[1] > (p_box[3] - (p_box[3] - p_box[1]) * Config.GROUND_FILTER_RATIO): continue

            vec_in = np.array(curr) - np.array(prev)
            vec_out = np.array(next_p) - np.array(curr)
            speed_in = np.linalg.norm(vec_in)
            speed_out = np.linalg.norm(vec_out)
            if speed_out < Config.MIN_SPEED_THRESH: continue
            denom = (speed_in * speed_out) + 1e-6
            cos_sim = np.dot(vec_in, vec_out) / denom
            is_angle = (cos_sim < Config.COS_SIM_THRESH)
            is_accel = (speed_out / (speed_in + 1e-6) > Config.ACCEL_RATIO_THRESH) and (speed_out > 8.0)
            if (is_angle or is_accel) and (i - last_frame > Config.MIN_HIT_GAP):
                hits.append({'frame': i, 'conf': 0.85 if is_accel else abs(cos_sim)})
                last_frame = i

        segments = []
        for h in hits:
            f = h['frame']
            if not player_boxes[f]: continue
            p_c = get_center(player_boxes[f][0])
            start = max(0, f - 10)
            for k in range(f, max(0, f - 40), -1):
                if ball_centers[k] and np.linalg.norm(np.array(ball_centers[k]) - np.array(p_c)) > Config.ACTION_DISTANCE_THRESH:
                    start = k
                    break
            end = min(total_frames - 1, f + 15)
            for k in range(f, min(total_frames, f + 40)):
                if ball_centers[k] and np.linalg.norm(np.array(ball_centers[k]) - np.array(p_c)) > Config.ACTION_DISTANCE_THRESH:
                    end = k
                    break
            s_final = max(0, start - Config.EXTRA_PADDING_PRE)
            e_final = min(total_frames - 1, end + Config.EXTRA_PADDING_POST)
            segments.append({
                'contact_frame': f, 'start_frame': s_final, 'end_frame': e_final,
                'segment_frames': list(range(s_final, e_final + 1)), 'sequence_length': e_final - s_final + 1, 'confidence': h['conf']
            })
        if not segments: return []
        segments.sort(key=lambda x: x['start_frame'])
        merged = [segments[0]]
        for curr in segments[1:]:
            last = merged[-1]
            if curr['start_frame'] <= last['end_frame'] + 5:
                last['end_frame'] = max(last['end_frame'], curr['end_frame'])
                last['segment_frames'] = list(range(last['start_frame'], last['end_frame'] + 1))
                last['sequence_length'] = len(last['segment_frames'])
                if curr['confidence'] > last['confidence']:
                    last['contact_frame'] = curr['contact_frame']
                    last['confidence'] = curr['confidence']
            else: merged.append(curr)
        return merged

# ===================== 5. Classification =====================
class ShotClassifier:
    def __init__(self):
        print("\n[2/5] Initializing Classifier...")
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1)
        self.model = None
        if os.path.exists(Config.FUSED_MODEL) or os.path.exists(os.path.join(Config.MODEL_DIR, Config.FUSED_MODEL)):
            try:
                orig_path = Config.ORIGINAL_MODEL if os.path.exists(Config.ORIGINAL_MODEL) else os.path.join(Config.MODEL_DIR, Config.ORIGINAL_MODEL)
                fused_path = Config.FUSED_MODEL if os.path.exists(Config.FUSED_MODEL) else os.path.join(Config.MODEL_DIR, Config.FUSED_MODEL)
                orig = OriginalTennisLSTM()
                ckpt = torch.load(orig_path, map_location='cpu')
                orig.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
                self.model = KnowledgeFusedLSTM(orig)
                ckpt_f = torch.load(fused_path, map_location='cpu')
                self.model.load_state_dict(ckpt_f['model_state_dict'] if 'model_state_dict' in ckpt_f else ckpt_f)
                self.model.to(Config.DEVICE).eval()
                print("  ✓ Models loaded.")
            except Exception as e: print(f"❌ Model Error: {e}")
        else: print("❌ Models not found, skipping classification.")

    def run(self, video_path, segments):
        if not self.model or not segments: return []
        print("  - Classifying shots...")
        cap = cv2.VideoCapture(video_path)
        results = []
        for i, seg in enumerate(segments):
            kps_seq = []
            for f_idx in seg['segment_frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = self.pose.process(rgb)
                    if res.pose_landmarks:
                        kps_seq.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark])
                    else: kps_seq.append(np.zeros((33, 4)))
                else: kps_seq.append(np.zeros((33, 4)))
            inp = torch.FloatTensor(np.array(kps_seq)).unsqueeze(0).to(Config.DEVICE)
            slen = torch.tensor([len(kps_seq)]).to(Config.DEVICE)
            with torch.no_grad():
                out = self.model(inp, slen)
                probs = torch.nn.functional.softmax(out, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            results.append({
                'segment_idx': i, 'start_frame': seg['start_frame'], 'hit_frame': seg['contact_frame'],
                'end_frame': seg['end_frame'], 'shot_type_id': pred, 'shot_type_str': Config.CLASS_NAMES[pred],
                'confidence': float(probs[0, pred].item())
            })
            print(f"\r    Classified {i+1}/{len(segments)}", end="")
        cap.release()
        print("")
        return results

# ===================== 6. IO & Visualization (Direct MP4) =====================
def save_json(results):
    print(f"\n[3/5] Saving JSON Data...")
    ensure_dir(os.path.dirname(Config.OUTPUT_JSON))
    json_data = []
    for r in results:
        json_data.append({
            "event_id": r['segment_idx'] + 1, "frame_start": int(r['start_frame']),
            "frame_hit": int(r['hit_frame']), "frame_end": int(r['end_frame']),
            "shot_type": r['shot_type_str'], "confidence": float(f"{r['confidence']:.4f}")
        })
    with open(Config.OUTPUT_JSON, 'w') as f: json.dump(json_data, f, indent=4)
    print(f"  ✓ Saved to {Config.OUTPUT_JSON}")

class Visualizer:
    def __init__(self):
        self.ball_trail = deque(maxlen=8)

    def draw_frame_graphics(self, frame, idx, results_map, player_boxes, ball_centers, total_frames, w, h):
        """Draw graphics on frame"""
        # Safety checks
        curr_box = player_boxes[idx][0] if idx < len(player_boxes) and player_boxes[idx] else None
        curr_ball = ball_centers[idx] if idx < len(ball_centers) else None

        # 1. Ball
        if curr_ball:
            self.ball_trail.append(curr_ball)
            cv2.circle(frame, curr_ball, 6, (0, 255, 255), -1)
        else:
            self.ball_trail.append(None)

        for t in range(1, len(self.ball_trail)):
            if self.ball_trail[t-1] and self.ball_trail[t]:
                thickness = int(math.sqrt(t * 1.5))
                cv2.line(frame, self.ball_trail[t-1], self.ball_trail[t], (0, 200, 255), thickness)

        # 2. Player
        if curr_box:
            x1, y1, x2, y2 = map(int, curr_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
            cv2.putText(frame, "PLAYER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 2)

        # 3. Info
        zone_map, type_map = results_map
        if idx in zone_map:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 10)

        if idx in type_map:
            res = type_map[idx]
            text = res['shot_type_str']
            color = Config.CLASS_COLORS[res['shot_type_id']]
            font_scale = 2.0; thickness = 4
            (wt, ht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx, cy = w // 2, h // 2
            cv2.rectangle(frame, (cx-wt//2-20, cy-ht-20), (cx+wt//2+20, cy+20), (0,0,0), -1)
            cv2.putText(frame, text, (cx-wt//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            if idx == res['hit_frame']:
                cv2.putText(frame, "CONTACT", (cx-100, cy+80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        # 4. Progress
        if total_frames > 0:
            prog = int(w * (idx / total_frames))
            cv2.line(frame, (0, h-5), (prog, h-5), (0, 255, 0), 5)

        return frame

    def generate(self, video_path, results, player_boxes, ball_centers):
        print("\n[5/5] Generating Visualization (Direct MP4 Mode)...")
        ensure_dir(Config.OUTPUT_DIR)

        # 1. 准备读取
        if not os.path.exists(video_path): return
        cap = cv2.VideoCapture(video_path)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # [CRITICAL FIX]: Force even dimensions for MP4/H.264
        # 很多编码器不支持奇数宽高
        w = orig_w if orig_w % 2 == 0 else orig_w - 1
        h = orig_h if orig_h % 2 == 0 else orig_h - 1

        print(f"  - Input: {orig_w}x{orig_h} -> Output: {w}x{h} (Even fixed)")

        # 2. 设置输出 (MP4)
        out_path = Config.OUTPUT_VIDEO

        # 尝试使用 mp4v (最通用)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if not out.isOpened():
            print("❌ VideoWriter failed with 'mp4v'. Trying 'avc1'...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            if not out.isOpened():
                print("❌ Critical Error: Cannot create MP4 file.")
                return

        print(f"  - Writer Opened! Codec: mp4v/avc1. File: {out_path}")

        # 3. 准备数据映射
        hit_map = {r['hit_frame']: r for r in results}
        zone_map = {}; type_map = {}
        for r in results:
            for f in range(r['start_frame'], r['end_frame'] + 1):
                zone_map[f] = True; type_map[f] = r
        results_map = (zone_map, type_map)

        # 4. 逐帧处理
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # 如果原尺寸被修改了，这里必须裁剪
                if w != orig_w or h != orig_h:
                    frame = frame[0:h, 0:w] # Crop to even size

                # 绘图
                frame = self.draw_frame_graphics(frame, idx, results_map, player_boxes, ball_centers, total_frames, w, h)

                # 写入
                out.write(frame)

                idx += 1
                if idx % 50 == 0: print(f"\r    Rendering: {idx}/{total_frames}", end="")
        except Exception as e:
            print(f"\n❌ Error during writing: {e}")
        finally:
            cap.release()
            out.release()
            print(f"\n✅ Finished! Processed {idx} frames.")
            print(f"✅ Saved to: {out_path}")

# ===================== Main =====================
if __name__ == "__main__":
    if not os.path.exists(Config.INPUT_VIDEO):
        print(f"❌ Error: Video not found at {Config.INPUT_VIDEO}")
    else:
        # 1. Detection
        detector = AdvancedShotDetector()
        segments, frames, fps, p_boxes, b_centers = detector.process_video(Config.INPUT_VIDEO)

        # 🧹 Memory Cleanup
        print("  🧹 Cleaning up memory...")
        del frames
        gc.collect()

        if segments:
            # 2. Classification
            classifier = ShotClassifier()
            full_results = classifier.run(Config.INPUT_VIDEO, segments)

            # 3. Save JSON
            save_json(full_results)

            # 4. Visualization (Direct MP4)
            viz = Visualizer()
            viz.generate(Config.INPUT_VIDEO, full_results, p_boxes, b_centers)
        else:
            print("No shots detected.")