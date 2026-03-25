"""
运动动作分析系统
羽毛球训练动作分析与评估
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# 导出核心模块
from .core.motion_analyzer import MotionAnalyzer
from .core.video_processor import VideoProcessor, VideoConfig, VideoSource
from .core.feedback_system import FeedbackSystem
from .core.ai_analyzer import AISuggestionSystem, DeepSeekAIProcessor

__all__ = [
    "MotionAnalyzer",
    "VideoProcessor",
    "VideoConfig",
    "VideoSource",
    "FeedbackSystem",
    "AISuggestionSystem",
    "DeepSeekAIProcessor",
]