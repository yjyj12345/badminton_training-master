"""
Sports Analyzer Core Modules
"""

from .motion_analyzer import MotionAnalyzer, MotionFrame, JointAngle
from .feedback_system import FeedbackSystem, FeedbackLevel, FeedbackType, Suggestion, TrainingFeedback
from .video_processor import VideoProcessor, VideoConfig, VideoSource, VideoAnalyzer, StreamingServer
from .ai_analyzer import AISuggestionSystem, AIAnalysisRequest# 新增

__all__ = [
    'MotionAnalyzer',
    'MotionFrame',
    'JointAngle',
    'FeedbackSystem',
    'FeedbackLevel',
    'FeedbackType',
    'Suggestion',
    'TrainingFeedback',
    'VideoProcessor',
    'VideoConfig',
    'VideoSource',
    'VideoAnalyzer',
    'StreamingServer',
    'AISuggestionSystem',
    'AIAnalysisRequest'

]
import logging
logging.basicConfig(level=logging.INFO)