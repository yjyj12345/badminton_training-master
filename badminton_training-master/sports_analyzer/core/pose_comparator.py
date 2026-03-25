import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class PoseComparator:
    def __init__(self, standards_dir="data/standard", cache_dir="standards/cache"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.standards_dir = standards_dir
        self.cache_dir = cache_dir

        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.standard_features = {}
        self._load_standards()

    def _load_standards(self):
        """加载标准库（优先读取缓存的 .npy 文件，没有则处理视频）"""
        actions = ['Forehand', 'Backhand', 'Serve']  # 必须与文件名对应

        for action in actions:
            cache_path = os.path.join(self.cache_dir, f"{action}.npy")
            video_path = os.path.join(self.standards_dir, f"{action}.mp4")

            if os.path.exists(cache_path):
                print(f"⚡ [加载缓存] 标准动作: {action}")
                self.standard_features[action] = np.load(cache_path)
            elif os.path.exists(video_path):
                print(f"🔄 [首次处理] 正在提取标准动作特征: {action} ...")
                features = self.extract_sequence(video_path)
                if features is not None and len(features) > 0:
                    # 归一化并保存
                    norm_features = self._normalize_sequence(features)
                    np.save(cache_path, norm_features)
                    self.standard_features[action] = norm_features
                    print(f"✅ 已生成缓存: {cache_path}")
            else:
                print(f"⚠️ [警告] 缺少标准视频文件: {video_path}")

    def extract_sequence(self, video_path):
        """从视频中提取骨骼序列 (Frames, 33, 2)"""
        cap = cv2.VideoCapture(video_path)
        features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # MediaPipe处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                frame_kps = []
                for lm in results.pose_landmarks.landmark:
                    # 只取 x, y 坐标
                    frame_kps.append([lm.x, lm.y])
                features.append(frame_kps)

        cap.release()
        return np.array(features) if features else None

    def _normalize_sequence(self, kps_seq):
        """
        动作归一化 (核心算法)
        消除身高、站位、相机距离的影响
        """
        data = np.array(kps_seq)  # Shape: (Frames, 33, 2)
        normalized_seq = []

        for frame in data:
            # 1. 计算中心点 (左右髋部的中间) -> 索引 23, 24
            hip_center = (frame[23] + frame[24]) / 2.0

            # 2. 计算躯干尺度 (肩膀中心到髋部中心的距离) -> 索引 11, 12, 23, 24
            shoulder_center = (frame[11] + frame[12]) / 2.0
            torso_size = np.linalg.norm(shoulder_center - hip_center)

            # 防止除以0
            if torso_size == 0: torso_size = 1.0

            # 3. 执行归一化: (坐标 - 中心) / 尺度
            # 这样所有人的髋部都在 (0,0)，躯干长度都是 1
            norm_frame = (frame - hip_center) / torso_size
            normalized_seq.append(norm_frame)

        return np.array(normalized_seq)

    def calculate_angle(self, a, b, c):
        """计算三点夹角 (用于生成具体建议)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def compare(self, user_kps_seq, shot_type):
        """
        对比入口函数
        :param user_kps_seq: 用户骨骼序列 (未归一化)
        :param shot_type: 动作类型字符串 'Forehand'
        :return: (分数, 详细指标)
        """
        if shot_type not in self.standard_features:
            print(f"❌ 无法对比: 缺少 {shot_type} 的标准数据")
            return 0, {}

        # 1. 获取标准数据
        std_norm = self.standard_features[shot_type]

        # 2. 归一化用户数据
        user_norm = self._normalize_sequence(user_kps_seq)

        if len(user_norm) < 10:
            return 0, {}  # 动作太短

        # 3. DTW 核心计算
        # 将 (Frames, 33, 2) 展平为 (Frames, 66) 以便计算距离
        user_flat = user_norm.reshape(len(user_norm), -1)
        std_flat = std_norm.reshape(len(std_norm), -1)

        # fastdtw 返回总距离和对齐路径
        distance, path = fastdtw(user_flat, std_flat, dist=euclidean)

        # 4. 计算相似度分数 (0-100)
        # 这里的 0.5 是一个经验系数，用于调整严苛程度
        # 距离越小，相似度越高
        similarity = max(0, 100 * (1 - distance / (len(path) * 0.5)))

        # 5. 计算关键生物力学指标 (取用户动作中间帧)
        mid_idx = len(user_kps_seq) // 2
        mid_frame = user_kps_seq[mid_idx]  # 使用原始坐标算角度

        # 计算膝盖角度 (取左右较小值)
        r_knee = self.calculate_angle(mid_frame[24], mid_frame[26], mid_frame[28])
        l_knee = self.calculate_angle(mid_frame[23], mid_frame[25], mid_frame[27])

        metrics = {
            "knee_angle": int(min(r_knee, l_knee)),
            # 可以继续添加 手臂角度、躯干倾角等
        }

        return int(similarity), metrics


# 单元测试
if __name__ == "__main__":
    print("正在测试 PoseComparator...")
    comparator = PoseComparator()
    # 如果 standards 文件夹里有视频，这里应该会显示加载或处理日志