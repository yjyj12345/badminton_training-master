# core/feedback_system.py
"""
智能反馈系统
根据动作分析结果生成个性化训练建议
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import sys
from pathlib import Path
import logging

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# 简化导入，避免循环
class FeedbackLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    NEEDS_IMPROVEMENT = "needs_improvement"


class FeedbackType(Enum):
    POSTURE = "posture"
    TIMING = "timing"
    BALANCE = "balance"
    COORDINATION = "coordination"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"


@dataclass
class Suggestion:
    type: FeedbackType
    priority: int  # 1-5, 1最高
    title: str
    description: str
    visual_hint: Optional[str] = None
    drill_recommendation: Optional[str] = None


@dataclass
class TrainingFeedback:
    overall_score: float
    level: FeedbackLevel
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[Suggestion]
    progress_notes: Optional[str] = None


# 简化版 MotionFrame 类定义，避免导入问题
class MotionFrame:
    """简化的运动帧定义"""
    def __init__(self, timestamp, landmarks, joint_angles, score, issues):
        self.timestamp = timestamp
        self.landmarks = landmarks
        self.joint_angles = joint_angles
        self.score = score
        self.issues = issues


class FeedbackSystem:
    """智能反馈生成系统"""
    
    def __init__(self):
        self.feedback_history: List[TrainingFeedback] = []
        self.user_skill_level = "intermediate"
        
    def generate_feedback(self, motion_frames: List[MotionFrame]) -> TrainingFeedback:
        """生成综合训练反馈"""
        if not motion_frames:
            return self._generate_empty_feedback()
            
        # 计算整体评分
        overall_score = self._calculate_overall_score(motion_frames)
        level = self._determine_level(overall_score)
        
        # 分析优缺点
        strengths = self._analyze_strengths(motion_frames)
        weaknesses = self._analyze_weaknesses(motion_frames)
        
        # 生成改进建议
        suggestions = self._generate_suggestions(motion_frames, weaknesses)
        
        # 生成进度说明
        progress_notes = self._generate_progress_notes(overall_score)
        
        feedback = TrainingFeedback(
            overall_score=overall_score,
            level=level,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            progress_notes=progress_notes
        )
        
        self.feedback_history.append(feedback)
        return feedback
    
    def _calculate_overall_score(self, motion_frames: List[MotionFrame]) -> float:
        """计算整体分数"""
        if not motion_frames:
            return 0.0
            
        scores = [frame.score for frame in motion_frames if frame.score > 0]
        if not scores:
            return 0.0
            
        # 加权平均：最近的帧权重更高
        weights = np.linspace(0.5, 1.0, len(scores))
        weighted_score = np.average(scores, weights=weights)
        
        return min(max(weighted_score * 100, 0), 100)
    
    def _determine_level(self, score: float) -> FeedbackLevel:
        """确定表现级别"""
        if score >= 90:
            return FeedbackLevel.EXCELLENT
        elif score >= 75:
            return FeedbackLevel.GOOD
        elif score >= 60:
            return FeedbackLevel.FAIR
        else:
            return FeedbackLevel.NEEDS_IMPROVEMENT
    
    def _analyze_strengths(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析动作优点"""
        strengths = []
        
        # 统计各项指标
        angle_scores = []
        
        for frame in motion_frames:
            if hasattr(frame, 'joint_angles') and frame.joint_angles:
                # 角度准确性
                try:
                    angle_deviations = [angle.deviation for angle in frame.joint_angles]
                    angle_score = 1.0 - (np.mean(angle_deviations) / 45.0)  # 标准化到0-1
                    angle_scores.append(angle_score)
                except:
                    pass
        
        # 分析一致性
        if len(motion_frames) > 10:
            scores = [f.score for f in motion_frames[-10:]]
            consistency = 1.0 - np.std(scores)
            if consistency > 0.7:
                strengths.append("动作稳定性良好")

        # 生成优点描述
        if angle_scores and np.mean(angle_scores) > 0.8:
            strengths.append("关节角度控制准确")
            
        if not strengths:
            strengths.append("动作识别正常")

        return strengths[:3]
    
    def _analyze_weaknesses(self, motion_frames: List[MotionFrame]) -> List[str]:
        """分析动作缺点"""
        weaknesses = []
        
        if not motion_frames:
            return ["未检测到动作数据"]
        
        # 检查是否有问题
        for frame in motion_frames[-10:]:  # 检查最近10帧
            if hasattr(frame, 'issues') and frame.issues:
                for issue in frame.issues:
                    if issue and issue not in weaknesses:
                        weaknesses.append(issue)
                        if len(weaknesses) >= 3:  # 最多3个
                            return weaknesses
        
        if not weaknesses:
            weaknesses.append("动作基本标准，继续保持")
            
        return weaknesses[:3]
    
    def _generate_suggestions(self, motion_frames: List[MotionFrame], weaknesses: List[str]) -> List[Suggestion]:
        """生成改进建议"""
        suggestions = []
        
        for weakness in weaknesses:
            if "角度" in weakness or "偏差" in weakness:
                suggestions.append(Suggestion(
                    type=FeedbackType.POSTURE,
                    priority=1,
                    title="改善关节角度控制",
                    description="注意保持正确的关节角度范围，可以参考专业运动员的动作",
                    visual_hint="保持肘部角度在90-150度之间",
                    drill_recommendation="对着镜子练习标准动作，每天10分钟"
                ))
            elif "平衡" in weakness or "重心" in weakness:
                suggestions.append(Suggestion(
                    type=FeedbackType.BALANCE,
                    priority=2,
                    title="增强身体平衡性",
                    description="加强核心力量训练，保持身体稳定",
                    visual_hint="保持重心在双脚之间",
                    drill_recommendation="单腿站立训练，每侧30秒×3组"
                ))
            elif "流畅" in weakness:
                suggestions.append(Suggestion(
                    type=FeedbackType.COORDINATION,
                    priority=2,
                    title="提高动作流畅性",
                    description="放松肌肉，保持动作连贯",
                    visual_hint="动作要连贯，避免停顿",
                    drill_recommendation="慢动作分解练习，每个动作重复20次"
                ))
        
        # 如果还没有建议，添加通用建议
        if not suggestions:
            suggestions.append(Suggestion(
                type=FeedbackType.POSTURE,
                priority=3,
                title="保持当前训练",
                description="继续按照当前计划训练，保持动作质量",
                visual_hint="注意呼吸和放松",
                drill_recommendation="常规训练，每周3-4次"
            ))
        
        return suggestions[:3]
    
    def _generate_progress_notes(self, current_score: float) -> str:
        """生成进度说明"""
        if not self.feedback_history:
            return "首次训练，继续保持！"
            
        if len(self.feedback_history) > 1:
            previous_score = self.feedback_history[-2].overall_score
            improvement = current_score - previous_score
            
            if improvement > 5:
                return f"进步明显！提升了{improvement:.1f}分"
            elif improvement > 0:
                return "稳步提升，继续努力！"
        
        return "表现稳定，继续保持"
    
    def _generate_empty_feedback(self) -> TrainingFeedback:
        """生成空反馈"""
        return TrainingFeedback(
            overall_score=0.0,
            level=FeedbackLevel.NEEDS_IMPROVEMENT,
            strengths=[],
            weaknesses=["未检测到有效动作"],
            suggestions=[
                Suggestion(
                    type=FeedbackType.POSTURE,
                    priority=1,
                    title="调整站位",
                    description="请确保全身在摄像头视野内",
                    visual_hint="站在摄像头前2-3米处"
                )
            ],
            progress_notes="等待动作检测..."
        )