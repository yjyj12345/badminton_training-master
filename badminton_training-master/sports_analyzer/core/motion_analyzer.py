# core/motion_analyzer.py
"""
网球动作评分模块
使用MediaPipe分析网球动作，提供各个维度的评分
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class JointAngle:
    """关节角度数据"""
    joint_name: str
    angle: float
    optimal_range: Tuple[float, float]
    deviation: float


@dataclass
class MotionFrame:
    """动作帧数据"""
    timestamp: float
    landmarks: np.ndarray
    joint_angles: List[JointAngle]
    score: float
    issues: List[str]


@dataclass
class DimensionScores:
    """各维度评分"""
    angle_control: float  # 角度控制 (0-100)
    balance: float  # 身体平衡 (0-100)
    fluidity: float  # 动作流畅度 (0-100)
    symmetry: float  # 左右对称性 (0-100)
    overall: float  # 综合评分 (0-100)


class MotionAnalyzer:
    """网球动作评分器"""

    def __init__(self):
        """初始化网球动作评分器"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_history: List[MotionFrame] = []
        self.session_joint_history: List[Dict] = []
        self.start_time = None

        logger.info("网球动作评分器初始化完成")

    def start_session(self):
        """开始新的训练会话"""
        self.session_joint_history.clear()
        self.start_time = time.time()
        self.frame_history.clear()
        logger.info("网球训练评分会话开始")

    def _save_joint_data(self, joint_angles: List[JointAngle], timestamp: float):
        """保存关节数据到会话历史"""
        frame_data = {
            "timestamp": timestamp,
            "joints": []
        }

        for angle in joint_angles:
            frame_data["joints"].append({
                "joint_name": angle.joint_name,
                "angle": float(angle.angle),
                "optimal_range": (float(angle.optimal_range[0]), float(angle.optimal_range[1])),
                "deviation": float(angle.deviation)
            })

        self.session_joint_history.append(frame_data)

        # 限制历史记录大小
        if len(self.session_joint_history) > 500:
            self.session_joint_history.pop(0)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Optional[MotionFrame]:
        """处理单帧图像并评分"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        landmarks = self._extract_landmarks(results.pose_landmarks)
        joint_angles = self._calculate_joint_angles(landmarks)
        score, issues = self._evaluate_pose(landmarks, joint_angles)

        motion_frame = MotionFrame(
            timestamp=timestamp,
            landmarks=landmarks,
            joint_angles=joint_angles,
            score=score,
            issues=issues
        )

        self._save_joint_data(joint_angles, timestamp)
        self.frame_history.append(motion_frame)

        # 保留最近300帧
        if len(self.frame_history) > 300:
            self.frame_history.pop(0)

        return motion_frame

    def _extract_landmarks(self, pose_landmarks) -> np.ndarray:
        """提取关键点坐标"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks)

    def _calculate_joint_angles(self, landmarks: np.ndarray) -> List[JointAngle]:
        """计算关节角度"""
        angles = []

        # 网球动作的关键关节连接
        joint_connections = {
            'right_elbow': (12, 14, 16),  # 主要击球手肘部
            'right_shoulder': (14, 12, 24),  # 击球手肩膀
            'right_hip': (12, 24, 26),  # 击球手髋部
            'right_knee': (24, 26, 28),  # 击球手膝盖
            'left_elbow': (11, 13, 15),  # 辅助手臂肘部
            'left_shoulder': (13, 11, 23),  # 辅助肩膀
            'left_hip': (11, 23, 25),  # 辅助髋部
            'left_knee': (23, 25, 27),  # 辅助膝盖
        }

        # 网球的最优角度范围
        optimal_angles = {
            'right_elbow': (80, 140),  # 发球和击球时
            'right_shoulder': (60, 110),  # 保持肩膀放松
            'right_hip': (85, 145),  # 重心转移
            'right_knee': (110, 160),  # 击球时微弯
            'left_elbow': (80, 140),  # 平衡手臂
            'left_shoulder': (60, 110),  # 辅助手臂
            'left_hip': (85, 145),  # 稳定性
            'left_knee': (110, 160),  # 准备姿势
        }

        for joint_name, (p1, p2, p3) in joint_connections.items():
            angle = self._calculate_angle(
                landmarks[p1][:3],
                landmarks[p2][:3],
                landmarks[p3][:3]
            )

            optimal_range = optimal_angles.get(joint_name, (0, 180))
            deviation = self._calculate_deviation(angle, optimal_range)

            angles.append(JointAngle(
                joint_name=joint_name,
                angle=angle,
                optimal_range=optimal_range,
                deviation=deviation
            ))

        return angles

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算三点形成的角度"""
        v1 = p1 - p2
        v2 = p3 - p2

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def _calculate_deviation(self, angle: float, optimal_range: Tuple[float, float]) -> float:
        """计算角度偏差"""
        if optimal_range[0] <= angle <= optimal_range[1]:
            return 0.0
        elif angle < optimal_range[0]:
            return optimal_range[0] - angle
        else:
            return angle - optimal_range[1]

    def _evaluate_pose(self, landmarks: np.ndarray, joint_angles: List[JointAngle]) -> Tuple[float, List[str]]:
        """评估姿势质量并计算各维度评分"""
        issues = []

        # 计算各维度评分
        angle_score = self._evaluate_angles(joint_angles, issues)
        balance_score = self._evaluate_balance(landmarks, issues)
        fluidity_score = self._evaluate_fluidity(landmarks, issues)
        symmetry_score = self._evaluate_symmetry(landmarks, joint_angles, issues)

        # 综合评分（加权平均）
        weights = [0.4, 0.3, 0.2, 0.1]  # 角度控制40%，平衡30%，流畅度20%，对称性10%
        dimension_scores = [angle_score, balance_score, fluidity_score, symmetry_score]
        total_score = sum(score * weight for score, weight in zip(dimension_scores, weights))

        # 转换为0-100分制
        total_score *= 100

        return total_score, issues

    def _evaluate_angles(self, joint_angles: List[JointAngle], issues: List[str]) -> float:
        """评估关节角度控制（0-1分）"""
        if not joint_angles:
            return 0.0

        scores = []
        for angle in joint_angles:
            if angle.deviation == 0:
                scores.append(1.0)
            elif angle.deviation < 5:
                scores.append(0.95)
            elif angle.deviation < 10:
                scores.append(0.85)
            elif angle.deviation < 15:
                scores.append(0.7)
                issues.append(f"{angle.joint_name}角度偏差{angle.deviation:.1f}°")
            elif angle.deviation < 20:
                scores.append(0.5)
                issues.append(f"{angle.joint_name}角度偏差较大{angle.deviation:.1f}°")
            else:
                scores.append(0.3)
                issues.append(f"{angle.joint_name}角度严重偏差{angle.deviation:.1f}°")

        return np.mean(scores) if scores else 0.0

    def _evaluate_balance(self, landmarks: np.ndarray, issues: List[str]) -> float:
        """评估身体平衡（0-1分）"""
        # 计算重心位置（髋部中心）
        hip_center = (landmarks[23][:2] + landmarks[24][:2]) / 2
        ankle_center = (landmarks[27][:2] + landmarks[28][:2]) / 2

        # 重心偏移量
        offset = np.linalg.norm(hip_center - ankle_center)

        if offset < 0.05:  # 优秀
            return 1.0
        elif offset < 0.1:  # 良好
            return 0.9
        elif offset < 0.15:  # 合格
            return 0.75
        elif offset < 0.2:  # 需改进
            issues.append("身体重心偏移")
            return 0.6
        else:  # 需加强
            issues.append("身体平衡性不佳")
            return 0.4

    def _evaluate_fluidity(self, landmarks: np.ndarray, issues: List[str]) -> float:
        """评估动作流畅度（0-1分）"""
        if len(self.frame_history) < 10:
            return 1.0  # 数据不足，给默认分

        # 计算关键点速度变化
        recent_frames = self.frame_history[-10:]
        velocities = []

        for i in range(1, len(recent_frames)):
            prev = recent_frames[i - 1].landmarks
            curr = recent_frames[i].landmarks
            dt = recent_frames[i].timestamp - recent_frames[i - 1].timestamp

            if dt > 0:
                velocity = np.linalg.norm(curr - prev) / dt
                velocities.append(velocity)

        if not velocities:
            return 1.0

        # 计算速度变化的标准差
        velocity_std = np.std(velocities)

        if velocity_std < 0.3:  # 非常流畅
            return 1.0
        elif velocity_std < 0.6:  # 流畅
            return 0.85
        elif velocity_std < 1.0:  # 一般
            return 0.7
        elif velocity_std < 1.5:  # 略有卡顿
            issues.append("动作流畅度有待提高")
            return 0.55
        else:  # 不流畅
            issues.append("动作不够流畅")
            return 0.4

    def _evaluate_symmetry(self, landmarks: np.ndarray, joint_angles: List[JointAngle], issues: List[str]) -> float:
        """评估左右对称性（0-1分）"""
        symmetry_pairs = [
            ('left_elbow', 'right_elbow'),
            ('left_shoulder', 'right_shoulder'),
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip')
        ]

        angle_dict = {angle.joint_name: angle.angle for angle in joint_angles}
        differences = []

        for left, right in symmetry_pairs:
            if left in angle_dict and right in angle_dict:
                diff = abs(angle_dict[left] - angle_dict[right])
                differences.append(diff)

        if not differences:
            return 1.0

        avg_diff = np.mean(differences)

        if avg_diff < 5:  # 高度对称
            return 1.0
        elif avg_diff < 10:  # 对称
            return 0.9
        elif avg_diff < 15:  # 基本对称
            return 0.75
        elif avg_diff < 20:  # 不对称
            issues.append("左右动作对称性需要改善")
            return 0.6
        else:  # 严重不对称
            issues.append("左右动作严重不对称")
            return 0.4

    def get_current_scores(self) -> Dict[str, Any]:
        """获取当前帧的各维度评分"""
        if not self.frame_history:
            return {
                "angle_control": 0,
                "balance": 0,
                "fluidity": 0,
                "symmetry": 0,
                "overall": 0,
                "frame_count": 0
            }

        latest_frame = self.frame_history[-1]

        # 重新计算各维度评分（确保一致性）
        issues = []
        angle_score = self._evaluate_angles(latest_frame.joint_angles, issues) * 100
        balance_score = self._evaluate_balance(latest_frame.landmarks, issues) * 100
        fluidity_score = self._evaluate_fluidity(latest_frame.landmarks, issues) * 100
        symmetry_score = self._evaluate_symmetry(latest_frame.landmarks, latest_frame.joint_angles, issues) * 100

        # 综合评分
        overall_score = latest_frame.score

        return {
            "angle_control": round(angle_score, 1),
            "balance": round(balance_score, 1),
            "fluidity": round(fluidity_score, 1),
            "symmetry": round(symmetry_score, 1),
            "overall": round(overall_score, 1),
            "frame_count": len(self.frame_history),
            "issues": issues[:3]  # 只返回前3个问题
        }

    def get_session_statistics(self) -> Dict[str, Any]:
        """获取整个会话的统计信息"""
        if not self.frame_history:
            return {
                "total_frames": 0,
                "duration": 0,
                "avg_overall_score": 0,
                "best_score": 0,
                "worst_score": 0
            }

        scores = [frame.score for frame in self.frame_history]
        duration = 0
        if self.session_joint_history and len(self.session_joint_history) > 1:
            duration = self.session_joint_history[-1]["timestamp"] - self.session_joint_history[0]["timestamp"]

        return {
            "total_frames": len(self.frame_history),
            "duration": round(duration, 2),
            "avg_overall_score": round(np.mean(scores), 1) if scores else 0,
            "best_score": round(np.max(scores), 1) if scores else 0,
            "worst_score": round(np.min(scores), 1) if scores else 0,
            "joints_analyzed": len(self._get_tennis_joints())
        }

    def _get_tennis_joints(self) -> List[str]:
        """获取网球分析的关键关节"""
        return [
            "right_elbow", "right_shoulder", "right_hip", "right_knee",
            "left_elbow", "left_shoulder", "left_hip", "left_knee"
        ]

    def get_joint_angle_data(self) -> List[Dict]:
        """获取关节角度数据（用于AI分析）"""
        if not self.frame_history:
            return []

        latest_frame = self.frame_history[-1]
        joint_data = []

        for angle in latest_frame.joint_angles:
            joint_data.append({
                "joint_name": angle.joint_name,
                "angle": round(angle.angle, 1),
                "optimal_min": angle.optimal_range[0],
                "optimal_max": angle.optimal_range[1],
                "deviation": round(angle.deviation, 1),
                "in_range": angle.deviation == 0
            })

        return joint_data

    def get_ai_ready_joint_data(self) -> List[Dict]:
        """获取用于AI分析的关节数据摘要"""
        if not self.frame_history:
            return []

        latest_frame = self.frame_history[-1] if self.frame_history else None
        if not latest_frame:
            return []

        joint_data = []
        for angle in latest_frame.joint_angles:
            joint_data.append({
                "joint_name": angle.joint_name,
                "angle": float(angle.angle),
                "deviation": float(angle.deviation)
            })

        return joint_data

    def get_recent_joint_data(self, num_frames: int = 30) -> List[Dict]:
        """获取最近的关节数据"""
        if not self.frame_history:
            return []

        frames = self.frame_history[-num_frames:] if len(self.frame_history) > num_frames else self.frame_history
        joint_data = []

        for frame in frames:
            for angle in frame.joint_angles:
                joint_data.append({
                    "joint_name": angle.joint_name,
                    "angle": float(angle.angle),
                    "deviation": float(angle.deviation),
                    "timestamp": frame.timestamp
                })

        return joint_data

    def get_joint_data_for_ai(self, num_frames: int = 1) -> List[Dict]:
        """获取用于AI分析的关节数据（主接口）"""
        """
        获取用于AI分析的关节数据

        Args:
            num_frames: 获取的帧数，1表示只获取最新一帧

        Returns:
            关节数据列表
        """
        if num_frames == 1:
            return self.get_ai_ready_joint_data()
        else:
            return self.get_recent_joint_data(num_frames)

    def draw_skeleton(self, frame: np.ndarray, motion_frame: MotionFrame) -> np.ndarray:
        """在图像上绘制骨架和评分"""
        if motion_frame is None:
            return frame

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # 绘制关键点
        for i, landmark in enumerate(motion_frame.landmarks):
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

        # 绘制骨架连接
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            start = motion_frame.landmarks[start_idx]
            end = motion_frame.landmarks[end_idx]

            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))

            cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)

        # 显示综合评分
        score_text = f"综合评分: {motion_frame.score:.1f}"
        cv2.putText(annotated_frame, score_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示关键问题
        y_offset = 60
        for issue in motion_frame.issues[:3]:  # 最多显示3个问题
            cv2.putText(annotated_frame, issue, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y_offset += 25

        # 在右上角显示各维度评分
        scores = self.get_current_scores()
        dimension_texts = [
            f"角度控制: {scores['angle_control']}",
            f"身体平衡: {scores['balance']}",
            f"动作流畅: {scores['fluidity']}",
            f"左右对称: {scores['symmetry']}"
        ]

        y_offset_right = 30
        for text in dimension_texts:
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            x_pos = w - text_width - 10
            cv2.putText(annotated_frame, text, (x_pos, y_offset_right),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset_right += 25

        return annotated_frame

    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        if not self.frame_history:
            return {
                "frame_count": 0,
                "duration": 0,
                "avg_score": 0
            }

        # 计算时长
        duration = 0
        if self.start_time:
            duration = time.time() - self.start_time

        # 计算平均分
        scores = [frame.score for frame in self.frame_history]
        avg_score = np.mean(scores) * 100 if scores else 0

        return {
            "frame_count": len(self.frame_history),
            "duration": duration,
            "avg_score": avg_score
        }

    def get_top_issues(self, limit: int = 3) -> List[str]:
        """获取主要问题"""
        if not self.frame_history:
            return []

        issue_counts = {}
        for frame in self.frame_history[-50:]:  # 检查最近50帧
            for issue in frame.issues:
                if issue in issue_counts:
                    issue_counts[issue] += 1
                else:
                    issue_counts[issue] = 1

        # 按频率排序
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:limit]]

    def reset(self):
        """重置评分器状态"""
        self.frame_history.clear()
        self.session_joint_history.clear()
        self.start_time = None
        logger.info("评分器已重置")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pose'):
            self.pose.close()