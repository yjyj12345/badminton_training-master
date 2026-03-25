# core/similarity.py
import numpy as np
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class ActionComparator:
    def __init__(self, standards_dir="data/standards"):
        self.standards_dir = standards_dir
        # Data structure: {'Forehand': [seq1, seq2...], 'Backhand': [...]}
        self.standards_map = {}
        self._load_standards()

    def _load_standards(self):
        """
        加载所有标准动作文件，并按动作类型分组
        确保加载的数据被切片到 (Frames, 33, 2)
        """
        self.standards_map = {'Forehand': [], 'Backhand': [], 'Serve': []}

        if not os.path.exists(self.standards_dir):
            print(f"❌ [对比器] 错误: 标准库目录不存在 -> {self.standards_dir}")
            return

        loaded_count = 0
        for f in os.listdir(self.standards_dir):
            if not f.endswith('.npy'): continue

            action_type = None
            if 'Forehand' in f:
                action_type = 'Forehand'
            elif 'Backhand' in f:
                action_type = 'Backhand'
            elif 'Serve' in f:
                action_type = 'Serve'

            if action_type:
                try:
                    path = os.path.join(self.standards_dir, f)
                    data = np.load(path)

                    # 1. 强制切片到 X, Y 坐标 (Frames, 33, 2)
                    if len(data.shape) == 3:
                        data = data[:, :, :2].astype(np.float32)

                    # 2. 存储归一化后的数据
                    norm_data = self._normalize_sequence(data)

                    self.standards_map[action_type].append({
                        "filename": f,
                        "data": data,
                        "norm_data": norm_data
                    })
                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️ 加载 {f} 失败: {e}")

        print(f"✅ [对比器] 标准库加载完毕，共 {loaded_count} 个参考样本。")

    def _normalize_sequence(self, kps_seq):
        """动作归一化 (消除身高、站位差异)"""
        data = np.array(kps_seq)
        if len(data.shape) == 3 and data.shape[2] > 2:
            data = data[:, :, :2]

        normalized_seq = []
        for frame in data:
            frame_xy = frame
            hip_center = (frame_xy[23] + frame_xy[24]) / 2.0
            shoulder_center = (frame_xy[11] + frame_xy[12]) / 2.0
            torso_size = np.linalg.norm(shoulder_center - hip_center)
            if torso_size == 0: torso_size = 1.0

            norm_frame = (frame - hip_center) / torso_size
            normalized_seq.append(norm_frame)

        return np.array(normalized_seq)

    def calculate_angle(self, a, b, c):
        """
        [修复点 1] 计算三点夹角 (b为顶点) - 确保输入是 (x, y) 向量
        """
        # 强制转换并确保只使用前两个维度（防止传入 [x, y, z, vis] 向量）
        a, b, c = np.array(a), np.array(b), np.array(c)
        a, b, c = a[:2], b[:2], c[:2]

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba * norm_bc == 0: return 0.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def get_frame_angles(self, frame):
        """计算单帧的8大关键关节角度"""
        # MediaPipe 索引映射
        joints = {
            "right_knee": (24, 26, 28),  # 右髋-右膝-右踝
            "left_knee": (23, 25, 27),  # 左髋-左膝-左踝
            "right_elbow": (12, 14, 16),  # 右肩-右肘-右腕
            "left_elbow": (11, 13, 15),  # 左肩-左肘-左腕
            "right_shoulder": (14, 12, 24),  # 右肘-右肩-右髋 (腋下开合度)
            "left_shoulder": (13, 11, 23),  # 左肘-左肩-左髋
            "right_hip": (12, 24, 26),  # 右肩-右髋-右膝 (躯干折叠度)
            "left_hip": (11, 23, 25)  # 左肩-左髋-左膝
        }

        angles = {}
        for name, (a, b, c) in joints.items():
            angles[name] = self.calculate_angle(frame[a], frame[b], frame[c])
        return angles

    def compare(self, user_kps_seq, shot_type):
        """
        执行多模态最佳匹配对比
        :return: (最佳相似度, 描述文本, 详细关节报告)
        """
        # 1. 基础检查
        candidates = self.standards_map.get(shot_type, [])
        if not candidates:
            return 0.0, f"标准库中缺少 {shot_type} 数据", []

        if not user_kps_seq or len(user_kps_seq) < 5:
            return 0.0, "用户动作数据不足", []

        # 2. 归一化用户数据
        # [修复点 2] 确保对用户数据进行归一化，且数据是 2D
        # 注意: user_kps_seq 可能来自 tennis_system (33, 4)，需要先切片
        user_kps_2d = np.array(user_kps_seq)
        if user_kps_2d.ndim == 3 and user_kps_2d.shape[2] > 2:
            user_kps_2d = user_kps_2d[:, :, :2]

        user_norm = self._normalize_sequence(user_kps_2d)
        user_flat = user_norm.reshape(len(user_norm), -1)

        # 3. 寻找最佳匹配 (Best Match)
        best_score = -1
        best_path = None
        best_std_norm = None

        for candidate in candidates:
            std_norm = candidate['norm_data']
            std_flat = std_norm.reshape(len(std_norm), -1)

            distance, path = fastdtw(user_flat, std_flat, dist=euclidean)
            current_score = max(0, 100 * (1 - distance / (len(path) * 0.5)))

            if current_score > best_score:
                best_score = current_score
                best_path = path
                best_std_norm = std_norm

        # 4. 基于最佳匹配路径，计算详细关节偏差
        if best_path is None:
            return 0.0, "对比失败", []

        joint_diff_sums = {}
        joint_counts = {}

        # 获取击球瞬间快照 (用户动作的中间帧)
        user_mid_idx = len(user_kps_seq) // 2
        snapshot_u_angles = {}
        snapshot_s_angles = {}

        # 沿 DTW 路径遍历每一帧
        for u_idx, s_idx in best_path:
            u_frame = user_norm[u_idx]
            s_frame = best_std_norm[s_idx]

            u_angles = self.get_frame_angles(u_frame)
            s_angles = self.get_frame_angles(s_frame)

            # 记录快照
            if u_idx == user_mid_idx:
                snapshot_u_angles = u_angles
                snapshot_s_angles = s_angles

            # 累积偏差
            for name, val in u_angles.items():
                s_val = s_angles.get(name, 0)
                diff = abs(val - s_val)
                joint_diff_sums[name] = joint_diff_sums.get(name, 0.0) + diff
                joint_counts[name] = joint_counts.get(name, 0) + 1

        # 5. 生成最终报告
        joint_details = []
        focus_prefix = "right" if "Forehand" in shot_type or "Serve" in shot_type else "left"

        for joint_name, total_diff in joint_diff_sums.items():
            count = joint_counts.get(joint_name, 1)
            avg_deviation = total_diff / count

            snap_u = snapshot_u_angles.get(joint_name, 0)
            snap_s = snapshot_s_angles.get(joint_name, 0)

            # 状态评级
            status = "完美"
            if avg_deviation > 20:
                status = "严重偏差"
            elif avg_deviation > 12:
                status = "需改进"
            elif avg_deviation > 7:
                status = "良好"

            # 方向判定 (基于击球帧)
            diff_snap = snap_u - snap_s
            direction = "过大/过直" if diff_snap > 0 else "过小/过弯"

            joint_details.append({
                "joint_name": joint_name,
                "angle": round(float(snap_u), 1),
                "pro_angle": round(float(snap_s), 1),
                "deviation": round(float(avg_deviation), 1),
                "direction": direction,
                "status": status,
                "is_key": focus_prefix in joint_name
            })

        feedback_text = f"匹配最佳样本 (相似度: {int(best_score)}%)"

        return best_score, feedback_text, joint_details