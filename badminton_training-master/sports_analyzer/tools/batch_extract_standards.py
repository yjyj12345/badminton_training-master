# tools/batch_extract_standards.py
import sys
import os
import shutil
import numpy as np
from pathlib import Path

# --- 1. 关键修复：环境与路径设置 ---
# 获取当前脚本所在目录 (project/tools)
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (project/)
project_root = current_dir.parent

# 将根目录加入系统路径，这样才能 import core
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心系统
try:
    from core.tennis_system import TennisSystem

    print("✅ 成功导入 core.tennis_system")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"   当前 sys.path: {sys.path}")
    sys.exit(1)

# --- 2. 配置参数 ---
# 强制使用绝对路径，防止“找不到文件”错误
RAW_DIR = os.path.join(project_root, "data", "raw_standards")
OUTPUT_DIR = os.path.join(project_root, "data", "standards")
MODEL_DIR = os.path.join(project_root, "models")  # 关键修复：指定模型绝对路径

# 质量门控
MIN_CONFIDENCE = 0.60  # 稍微降低一点门槛，防止漏掉好动作
MIN_FRAMES = 15  # 动作太短可能是误检


def batch_extract():
    # 检查必要文件夹
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 致命错误: 模型目录不存在 -> {MODEL_DIR}")
        return

    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        print(f"⚠️ 未找到原始视频目录，已自动创建: {RAW_DIR}")
        print("请把标准动作视频 (Forehand.mp4 等) 放入该文件夹后重试。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 初始化 AI 系统 (模型路径: {MODEL_DIR})...")
    try:
        # 关键修复：传入绝对路径的 model_dir
        system = TennisSystem(model_dir=MODEL_DIR)
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("可能是缺少 yolov8x.pt 或 best_model.pth，请检查 models 文件夹。")
        return

    # 获取所有视频文件
    video_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

    if not video_files:
        print(f"❌ 错误: {RAW_DIR} 文件夹是空的！")
        return

    print(f"📂 发现 {len(video_files)} 个视频文件，准备处理...")
    total_extracted = 0

    for vid_name in video_files:
        print(f"\n========================================")
        print(f"🎥 正在分析: {vid_name}")

        input_path = os.path.join(RAW_DIR, vid_name)
        # 临时输出视频路径 (tennis_system 需要这个参数，哪怕我们最后删掉它)
        temp_out_vid = os.path.join(RAW_DIR, f"temp_viz_{vid_name}")

        try:
            # 调用处理
            # 注意：这里必须传两个参数，因为你的 tennis_system.process_video 定义是 (input, output)
            result = system.process_video(input_path, temp_out_vid)

            # 获取击球数据
            shots = result.get('shots', [])
            print(f"📊 检测到 {len(shots)} 个潜在击球片段")

            saved_count = 0
            # 提取文件名作为前缀 (去除扩展名)
            base_name = os.path.splitext(vid_name)[0]

            for i, shot in enumerate(shots):
                shot_type = shot['type']  # e.g. "Forehand"
                confidence = shot['confidence']
                kps_seq = shot['kps_seq']  # 骨骼序列
                duration = len(kps_seq)

                # 质量过滤
                if confidence < MIN_CONFIDENCE:
                    print(f"  -> 忽略片段 {i + 1} (置信度 {confidence:.2f} 低于 {MIN_CONFIDENCE})")
                    continue

                if duration < MIN_FRAMES:
                    print(f"  -> 忽略片段 {i + 1} (帧数 {duration} 小于 {MIN_FRAMES})")
                    continue

                # 保存数据
                # 命名格式: 动作类型_来源视频_序号.npy
                save_name = f"{shot_type}_{base_name}_{i:03d}.npy"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                np.save(save_path, np.array(kps_seq))

                print(f"  -> ✅ 保存: {save_name} (Conf: {confidence:.2f})")
                saved_count += 1

            total_extracted += saved_count

        except Exception as e:
            print(f"❌ 处理视频 {vid_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理临时视频
            if os.path.exists(temp_out_vid):
                try:
                    os.remove(temp_out_vid)
                except:
                    pass

    print(f"\n========================================")
    print(f"✅ 全部完成！共提取 {total_extracted} 个标准动作文件。")
    print(f"📁 数据保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    batch_extract()