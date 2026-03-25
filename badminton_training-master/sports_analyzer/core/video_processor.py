"""
视频处理模块
处理实时视频流、文件上传和视频分析
"""
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import asyncio
import time
from typing import Optional, Callable, Any, Dict, List, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue
import base64
import os

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


if TYPE_CHECKING:
    from .motion_analyzer import MotionFrame

class VideoSource(Enum):
    FILE = "file"

@dataclass
class VideoConfig:
    source_type: VideoSource
    source_path: Optional[str] = None
    fps: int = 30
    resolution: tuple = (1280, 720)
    buffer_size: int = 10
    enable_recording: bool = False
    recording_path: Optional[str] = None

@dataclass
class ProcessedFrame:
    frame_id: int
    timestamp: float
    original_frame: np.ndarray
    processed_frame: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class VideoProcessor:
    """核心视频处理器类 - 处理视频流捕获和处理"""

    def __init__(self, config: VideoConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.processors: List[Callable] = []
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.stats = {
            "fps": 0,
            "processed_frames": 0,
            "dropped_frames": 0,
            "avg_processing_time": 0,
            "current_score": 0,
            "frame_count": 0,
            "actual_fps": 0
        }
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_id = 0
        self.last_frame_time = time.time()

    # video_processor.py 中的 initialize 方法修改：
    def initialize(self) -> bool:
        """初始化视频源"""
        try:
            if self.config.source_type == VideoSource.FILE:
                if not self.config.source_path:
                    raise ValueError("File source requires source_path")

                if not os.path.exists(self.config.source_path):
                    raise FileNotFoundError(f"视频文件不存在: {self.config.source_path}")

                self.cap = cv2.VideoCapture(self.config.source_path)

                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.config.source_path, cv2.CAP_FFMPEG)

            elif self.config.source_type == VideoSource.WEBCAM:
                self.cap = cv2.VideoCapture(0)
                if self.config.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                if self.config.fps:
                    self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            if not self.cap or not self.cap.isOpened():
                print("Failed to open video source")
                return False

            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"视频源信息: {actual_width}x{actual_height}, FPS: {actual_fps:.1f}")

            return True

        except Exception as e:
            print(f"Error initializing video source: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_processor(self, processor: Callable):
        """添加帧处理器"""
        self.processors.append(processor)

    def _process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        """处理单个帧"""
        start_time = time.time()

        frame_data = ProcessedFrame(
            frame_id=self.frame_id,
            timestamp=time.time(),
            original_frame=frame.copy()
        )

        for processor in self.processors:
            try:
                frame_data = processor(frame_data)
            except Exception as e:
                print(f"Processor error: {e}")

        processing_time = time.time() - start_time
        self.stats["avg_processing_time"] = (
                                                    self.stats["avg_processing_time"] * self.stats[
                                                "processed_frames"] + processing_time
                                            ) / (self.stats["processed_frames"] + 1)
        self.stats["processed_frames"] += 1

        return frame_data

    # video_processor.py 中的 _capture_thread 方法修改：
    def _capture_thread(self):
        """视频捕获线程"""
        frame_times = []
        print(f"开始视频捕获线程，源类型: {self.config.source_type}")

        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print("视频捕获对象未初始化或已关闭")
                    time.sleep(0.5)
                    continue

                ret, frame = self.cap.read()

                if not ret:
                    if self.config.source_type == VideoSource.FILE:
                        current_position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

                        if current_position >= total_frames - 1:
                            print("视频文件已播放完毕")
                            self.is_running = False
                            break

                    time.sleep(0.1)
                    continue

                if frame is None or frame.size == 0:
                    continue

                # 处理帧
                processed_frame = self._process_frame(frame)

                # 放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame)
                else:
                    self.stats["dropped_frames"] += 1

                # 计算FPS
                current_time = time.time()
                frame_times.append(current_time)

                if len(frame_times) > 30:
                    frame_times.pop(0)

                if len(frame_times) > 1:
                    time_span = frame_times[-1] - frame_times[0]
                    if time_span > 0:
                        self.stats["actual_fps"] = len(frame_times) / time_span
                        self.stats["fps"] = self.stats["actual_fps"]

                self.frame_id += 1
                self.stats["frame_count"] = self.frame_id

                # 如果是文件，控制播放速度
                if self.config.source_type == VideoSource.FILE:
                    time.sleep(1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30))
                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"捕获线程错误: {e}")
                time.sleep(0.1)

        print("视频捕获线程结束")

    def start(self):
        """开始视频处理"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._capture_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Video processor started")

    def stop(self):
        """停止视频处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("Video processor stopped")

    def get_frame_for_display(self) -> Optional[np.ndarray]:
        """获取用于显示的帧"""
        if not self.frame_queue.empty():
            try:
                frame_data = self.frame_queue.get_nowait()
                if frame_data.metadata and "display_frame" in frame_data.metadata:
                    return frame_data.metadata["display_frame"]
                return frame_data.processed_frame if frame_data.processed_frame is not None else frame_data.original_frame
            except:
                pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def get_frame(self) -> Optional[ProcessedFrame]:
        """获取处理后的帧（包含元数据）"""
        if not self.frame_queue.empty():
            try:
                return self.frame_queue.get_nowait()
            except:
                pass
        return None


class StreamingServer:
    """视频流服务器（用于Web端展示）"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.clients: List[asyncio.Queue] = []
        self.is_running = False

    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        queue = asyncio.Queue()
        self.clients.append(queue)

        try:
            while True:
                frame_data = await queue.get()
                await websocket.send(frame_data)
        except Exception as e:
            print(f"Client disconnected: {e}")
        finally:
            self.clients.remove(queue)

    async def broadcast_frame(self, frame: np.ndarray):
        """广播视频帧到所有客户端"""
        if not self.clients:
            return

        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # 发送到所有客户端
        for queue in self.clients:
            try:
                await queue.put(frame_data)
            except:
                pass

    async def start(self):
        """启动流服务器"""
        import websockets
        self.is_running = True
        await websockets.serve(self.handle_client, "localhost", self.port)
        print(f"Streaming server started on port {self.port}")

    def stop(self):
        """停止流服务器"""
        self.is_running = False


class VideoAnalyzer:
    """视频分析器（整合视频处理和动作分析）"""

    def __init__(self):
        self.video_processor = None
        self.motion_analyzer = None
        self.feedback_system = None
        self.streaming_server = None
        self.analysis_results = []
        self.current_frame_data = None  # 添加当前帧数据存储
        self.motion_frames = []
        self.joint_data_history = []

    def setup(self, video_config: VideoConfig):
        """设置分析器"""
        try:
            # 直接导入，不使用相对导入
            from .motion_analyzer import MotionAnalyzer
            from .feedback_system import FeedbackSystem

            # 初始化组件
            self.video_processor = VideoProcessor(video_config)
            self.motion_analyzer = MotionAnalyzer()
            self.feedback_system = FeedbackSystem()

            # 添加动作分析处理器
            self.video_processor.add_processor(self._analyze_motion)

            # 初始化视频源
            return self.video_processor.initialize()

        except Exception as e:
            print(f"Error in VideoAnalyzer setup: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _analyze_motion(self, frame_data: ProcessedFrame) -> ProcessedFrame:
        """分析动作并收集关节数据"""
        try:
            # 使用动作分析器处理帧
            motion_frame = self.motion_analyzer.process_frame(
                frame_data.original_frame,
                frame_data.timestamp
            )

            if motion_frame:
                # 绘制骨架
                annotated_frame = self.motion_analyzer.draw_skeleton(
                    frame_data.original_frame,
                    motion_frame
                )
                frame_data.processed_frame = annotated_frame

                # 保存元数据
                frame_data.metadata = {
                    "display_frame": annotated_frame,
                    "score": getattr(motion_frame, 'score', 0.0),
                    "issues": getattr(motion_frame, 'issues', []),
                    "joint_angles": []
                }

                # 获取关节角度信息
                if hasattr(motion_frame, 'joint_angles'):
                    joint_data = []
                    for angle in motion_frame.joint_angles:
                        joint_info = {
                            "joint_name": getattr(angle, 'joint_name', 'unknown'),
                            "angle": getattr(angle, 'angle', 0.0),
                            "deviation": getattr(angle, 'deviation', 0.0),
                            "optimal_range": getattr(angle, 'optimal_range', (0, 180))
                        }
                        frame_data.metadata["joint_angles"].append(joint_info)
                        joint_data.append(joint_info)

                    # 保存关节数据用于AI分析
                    self.joint_data_history.append({
                        "timestamp": frame_data.timestamp,
                        "joints": joint_data
                    })
                    # 限制历史记录大小
                    if len(self.joint_data_history) > 100:
                        self.joint_data_history.pop(0)

                # 更新统计数据
                self.video_processor.stats["current_score"] = motion_frame.score

                # 保存到列表
                self.motion_frames.append(motion_frame)
                self.current_frame_data = motion_frame

                if len(self.motion_frames) > 300:
                    self.motion_frames.pop(0)

        except Exception as e:
            print(f"Error analyzing motion: {e}")

        return frame_data

    def analyze_video_sync(self):
        """同步分析视频（用于API调用）"""
        if not self.video_processor:
            print("Video processor not initialized")
            return None

        print("开始同步视频分析...")
        self.video_processor.start()

        all_results = []

        # 主循环
        while self.video_processor.is_running:
            frame_data = self.video_processor.get_frame()

            if frame_data is None:
                # 短暂等待以避免CPU过载
                time.sleep(0.01)
                continue

            # 收集分析结果
            if frame_data.metadata:
                all_results.append({
                    "frame_id": frame_data.frame_id,
                    "timestamp": frame_data.timestamp,
                    "score": frame_data.metadata.get("score", 0),
                    "issues": frame_data.metadata.get("issues", []),
                    "joint_angles": frame_data.metadata.get("joint_angles", [])
                })

        # 停止处理
        self.video_processor.stop()

        # 生成反馈
        if self.motion_frames and self.feedback_system:
            try:
                feedback = self.feedback_system.generate_feedback(self.motion_frames)
                # 添加关节数据历史
                feedback.joint_data_history = self.joint_data_history
                feedback.analysis_results = all_results
                return feedback
            except Exception as e:
                print(f"Error generating feedback: {e}")
                import traceback
                traceback.print_exc()

        return None

    def get_joint_data_for_ai(self, recent_frames: int = 10) -> List[Dict]:
        """获取用于AI分析的关节数据"""
        if not self.joint_data_history:
            return []

        # 返回最近的关节数据
        recent_data = self.joint_data_history[-recent_frames:]

        # 提取并平均化数据
        averaged_joints = {}

        for data in recent_data:
            for joint in data["joints"]:
                joint_name = joint["joint_name"]
                if joint_name not in averaged_joints:
                    averaged_joints[joint_name] = {
                        "joint_name": joint_name,
                        "angles": [],
                        "deviations": []
                    }

                averaged_joints[joint_name]["angles"].append(joint["angle"])
                averaged_joints[joint_name]["deviations"].append(joint["deviation"])

        # 计算平均值
        result = []
        for joint_name, data in averaged_joints.items():
            if data["angles"]:
                result.append({
                    "joint_name": joint_name,
                    "angle": np.mean(data["angles"]),
                    "deviation": np.mean(data["deviations"])
                })

        return result

    def stop_analysis(self):
        """停止分析"""
        if self.video_processor:
            self.video_processor.stop()

        # 生成最终反馈
        if self.motion_frames and self.feedback_system:
            try:
                feedback = self.feedback_system.generate_feedback(self.motion_frames)
                return feedback
            except Exception as e:
                print(f"Error generating feedback: {e}")

        return None

    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        stats = {}

        if self.video_processor:
            stats = self.video_processor.get_stats()

        if self.motion_frames:
            recent_scores = [getattr(r, 'score', 0) for r in self.motion_frames[-30:]]
            if recent_scores:
                stats["avg_score"] = np.mean(recent_scores)
                stats["current_score"] = getattr(self.motion_frames[-1], 'score', 0) if self.motion_frames else 0
                stats["best_score"] = np.max(recent_scores)

        return stats

    # 添加前端需要的方法
    def get_motion_frames(self) -> List:
        """获取所有动作帧数据"""
        return self.motion_frames

    def get_current_frame_data(self):
        """获取当前帧数据"""
        return self.current_frame_data

    def get_analysis_results(self):
        """获取分析结果"""
        return self.analysis_results


# 使用示例
if __name__ == "__main__":
    # 配置视频源
    pass
