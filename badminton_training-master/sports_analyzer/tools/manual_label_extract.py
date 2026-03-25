# tools/single_video_extract.py
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# --- 1. 环境设置 (确保能导入 core) ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.tennis_system import TennisSystem, get_center, get_area
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= 配置区域 (修改这里) =================

# 1. 输入视频文件名 (必须位于 data/raw_standards 下)
VIDEO_FILENAME = "user9.mp4"

# 2. 强制指定的动作类型 (作为文件名前缀)
# 例如: 'Forehand', 'Backhand', 'Serve'
TARGET_SHOT_TYPE = "Backhand"

# 3. 质量控制
MIN_FRAMES = 5  # 太短的动作不要


# ======================================================

def extract_single_video():
    # 路径构建
    raw_dir = os.path.join(project_root, "data", "raw_standards")
    output_dir = os.path.join(project_root, "data", "standards")
    model_dir = os.path.join(project_root, "models")

    input_path = os.path.join(raw_dir, VIDEO_FILENAME)

    # 检查文件
    if not os.path.exists(input_path):
        print(f"❌ 找不到视频文件: {input_path}")
        print(f"请确认文件是否在 {raw_dir} 目录下。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化系统
    print(f"🚀 初始化系统 (处理单个视频: {VIDEO_FILENAME})...")
    try:
        system = TennisSystem(model_dir=model_dir)
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    print(f"\n========================================")
    print(f"🎥 开始分析: {VIDEO_FILENAME}")
    print(f"🏷️ 强制标签: {TARGET_SHOT_TYPE}")

    # --- 步骤 1: 读取视频 ---
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    if not frames:
        print("❌ 视频读取失败或为空")
        return

    # --- 步骤 2: 物理检测 (找切片) ---
    print(f"  -> 运行 YOLO 检测 ({len(frames)} 帧)...")
    player_boxes, ball_centers = system._run_yolo(frames)

    print("  -> 计算物理轨迹与切片...")
    ball_smooth = system._interpolate_ball(ball_centers)
    # 调用系统内部的物理逻辑找到击球片段
    segments = system._detect_physics_logic(ball_smooth, player_boxes, len(frames))

    print(f"📊 物理引擎识别出 {len(segments)} 个动作片段")

    # --- 步骤 3: 提取骨骼并保存 (跳过分类) ---
    base_name = os.path.splitext(VIDEO_FILENAME)[0]
    saved_count = 0

    # 重新打开视频用于提取特定帧
    cap = cv2.VideoCapture(input_path)

    for i, seg in enumerate(segments):
        # 提取骨骼数据 (复用 MediaPipe 逻辑)
        kps_seq = []
        for f_idx in seg['segment_frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = system.pose.process(rgb)
                if res.pose_landmarks:
                    # [x, y, z, visibility]
                    kps_seq.append([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark])
                else:
                    kps_seq.append(np.zeros((33, 4)).tolist())
            else:
                kps_seq.append(np.zeros((33, 4)).tolist())

        # 质量过滤
        duration = len(kps_seq)
        if duration < MIN_FRAMES:
            print(f"  -> ❌ 忽略片段 {i} (帧数 {duration} 过短)")
            continue

        # --- 核心：强制命名并保存 ---
        # 格式: {shot_type}_{base_name}_{i:03d}.npy
        save_name = f"{TARGET_SHOT_TYPE}_{base_name}_{i:03d}.npy"
        save_path = os.path.join(output_dir, save_name)

        np.save(save_path, np.array(kps_seq))
        print(f"  -> ✅ 保存: {save_name} (Frames: {duration})")

        saved_count += 1

    cap.release()

    print(f"\n========================================")
    print(f"🎉 处理完成！")
    print(f"📂 输入视频: {VIDEO_FILENAME}")
    print(f"💾 生成文件: {saved_count} 个")
    print(f"🏷️ 动作类型: {TARGET_SHOT_TYPE}")
    print(f"📁 保存路径: {output_dir}")


if __name__ == "__main__":
    extract_single_video()