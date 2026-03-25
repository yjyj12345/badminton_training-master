import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# ========== 1. 设置路径 ==========
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

print(f"当前文件: {current_file}")
print(f"项目根目录: {project_root}")
print(f"已将项目根目录添加到sys.path: {project_root}")

# ========== 2. 导入核心模块 ==========
print("\n开始导入核心模块...")


# 定义占位符类
class PlaceholderClass:
    def __init__(self, *args, **kwargs):
        pass


# 导入函数
def import_class(module_path: str, class_name: str):
    """导入单个类"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ 导入 {class_name}")
        return cls
    except (ImportError, AttributeError) as e:
        print(f"❌ 导入 {class_name} 失败: {e}")
        return None


# 导入所有需要的类
SiliconFlowAIProcessor = import_class("core.ai_analyzer", "SiliconFlowAIProcessor")
AISuggestionSystem = import_class("core.ai_analyzer", "AISuggestionSystem")
MotionAnalyzer = import_class("core.motion_analyzer", "MotionAnalyzer")
MotionFrame = import_class("core.motion_analyzer", "MotionFrame")
FeedbackSystem = import_class("core.feedback_system", "FeedbackSystem")
VideoAnalyzer = import_class("core.video_processor", "VideoAnalyzer")
VideoProcessor = import_class("core.video_processor", "VideoProcessor")
VideoConfig = import_class("core.video_processor", "VideoConfig")
VideoSource = import_class("core.video_processor", "VideoSource")

# --- [新增] 导入新的网球系统模块 (如果文件不存在，使用占位符防止报错) ---
try:
    from core.tennis_system import TennisSystem
    from core.similarity import ActionComparator

    print("✅ 导入 TennisSystem 和 ActionComparator")
except ImportError as e:
    print(f"⚠️ 导入网球高级模块失败: {e}，将不可用")
    TennisSystem = PlaceholderClass
    ActionComparator = PlaceholderClass

# 检查导入结果
print("\n导入状态总结:")
classes_to_check = [
    ("SiliconFlowAIProcessor", SiliconFlowAIProcessor),
    ("AISuggestionSystem", AISuggestionSystem),
    ("MotionAnalyzer", MotionAnalyzer),
    ("MotionFrame", MotionFrame),
    ("FeedbackSystem", FeedbackSystem),
    ("VideoAnalyzer", VideoAnalyzer),
    ("VideoProcessor", VideoProcessor),
    ("VideoConfig", VideoConfig),
    ("VideoSource", VideoSource),
]

# 为缺失的类创建占位符
for i, (name, cls) in enumerate(classes_to_check):
    if cls is None:
        print(f"⚠️  {name}: 导入失败，将使用占位符")
        placeholder = type(f"Placeholder{name}", (PlaceholderClass,), {})
        if name == "SiliconFlowAIProcessor":
            SiliconFlowAIProcessor = placeholder
        elif name == "AISuggestionSystem":
            AISuggestionSystem = placeholder
        elif name == "MotionAnalyzer":
            MotionAnalyzer = placeholder
        elif name == "MotionFrame":
            MotionFrame = placeholder
        elif name == "FeedbackSystem":
            FeedbackSystem = placeholder
        elif name == "VideoAnalyzer":
            VideoAnalyzer = placeholder
        elif name == "VideoProcessor":
            VideoProcessor = placeholder
        elif name == "VideoConfig":
            VideoConfig = placeholder
        elif name == "VideoSource":
            VideoSource = placeholder
    else:
        print(f"✅  {name}: 成功导入")

print("\n✅ 导入完成，准备启动FastAPI...")

# ========== 3. 导入FastAPI和其他依赖 ==========
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
import asyncio
import json
import uuid
import time
from datetime import datetime
import cv2
import base64
import io

active_sessions: Dict[str, Any] = {}
motion_sessions: Dict[str, Dict[str, Any]] = {}
ai_suggestion_system = AISuggestionSystem()

# --- [新增] 初始化网球专用系统 ---
# 确保在项目根目录下有 models 和 data/standards 文件夹
try:
    tennis_system = TennisSystem(model_dir="./models")
    comparator = ActionComparator(standards_dir="./data/standards")
    print("✅ TennisSystem 与 Comparator 初始化成功")
except Exception as e:
    print(f"⚠️ 初始化 TennisSystem 失败: {e}")
    tennis_system = None
    comparator = None

app = FastAPI(
    title="Sports Analyzer API",
    description="AI-powered sports training analysis system",
    version="1.0.0"
)

# ========== CORS 配置 ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 挂载前端页面 ==========
try:
    base_dir = Path(__file__).resolve().parent.parent.parent
    frontend_dir = base_dir / "frontend"
    if not frontend_dir.exists():
        frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

    if frontend_dir.exists():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        print(f"\n✅ 前端页面已挂载! 请访问: http://localhost:8000/frontend/index.html")
    else:
        print(f"\n⚠️ 未找到前端目录: {frontend_dir}")

    # [新增] 挂载结果目录，用于访问生成的分析视频
    os.makedirs("results", exist_ok=True)
    app.mount("/results", StaticFiles(directory="results"), name="results")

except Exception as e:
    print(f"\n❌ 挂载目录时出错: {e}")


# ================= 辅助类：全关节计算器 (新增) =================
class DetailedPoseCalculator:
    """专门用于计算全身关键关节角度，为了适配 AI 的提示词模板"""

    @staticmethod
    def calculate_angle(a, b, c):
        """计算三点夹角"""
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def get_full_body_metrics(user_kps, std_kps=None):
        """计算8大核心关节的角度"""
        # 关节索引映射 (A, B, C) -> B是顶点 (MediaPipe索引)
        joints_map = {
            "right_knee": (24, 26, 28),  # 右膝 (髋-膝-踝)
            "left_knee": (23, 25, 27),  # 左膝
            "right_elbow": (12, 14, 16),  # 右肘 (肩-肘-腕)
            "left_elbow": (11, 13, 15),  # 左肘
            "right_shoulder": (14, 12, 24),  # 右肩 (肘-肩-髋)
            "left_shoulder": (13, 11, 23),  # 左肩
            "right_hip": (12, 24, 26),  # 右髋 (肩-髋-膝)
            "left_hip": (11, 23, 25)  # 左髋
        }

        joint_data = []
        # 取中间帧
        u_frame = user_kps[len(user_kps) // 2]
        s_frame = std_kps[len(std_kps) // 2] if std_kps is not None else None

        for name, (idx_a, idx_b, idx_c) in joints_map.items():
            try:
                u_angle = DetailedPoseCalculator.calculate_angle(u_frame[idx_a], u_frame[idx_b], u_frame[idx_c])
                deviation = 0.0
                if s_frame is not None:
                    s_angle = DetailedPoseCalculator.calculate_angle(s_frame[idx_a], s_frame[idx_b], s_frame[idx_c])
                    deviation = abs(u_angle - s_angle)

                joint_data.append({
                    "joint_name": name,
                    "angle": float(u_angle),
                    "deviation": float(deviation)
                })
            except Exception:
                continue  # 防止索引越界或数据错误

        return joint_data


# ================= Pydantic模型 (保持原有) =================
class SessionConfig(BaseModel):
    video_source: str = "webcam"
    enable_recording: bool = False
    user_id: Optional[str] = None


class FrameAnalysis(BaseModel):
    timestamp: float
    score: float
    issues: List[str]
    suggestions: Optional[List[str]] = None


class AIFeedbackRequest(BaseModel):
    session_id: str
    user_level: str = "intermediate"
    include_detailed_analysis: bool = True


class AIFeedbackResponse(BaseModel):
    overall_score: float
    level: str
    dimensions: Dict[str, float]
    strengths: List[str]
    improvements: List[str]
    suggestions: Dict[str, Any]
    training_tips: List[Dict[str, Any]]
    action_plan: Dict[str, Any]


class TrainingSession(BaseModel):
    session_id: str
    user_id: Optional[str]
    sport_type: str
    start_time: datetime
    duration: float
    avg_score: float
    total_frames: int


class FeedbackResponse(BaseModel):
    overall_score: float
    level: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[Dict[str, str]]
    progress_notes: Optional[str]


# ================= API端点 =================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Sports Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "session": "/api/v1/session",
            "analysis": "/api/v1/analysis",
            "frontend": "/frontend/index.html"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/session/start")
async def start_session(config: SessionConfig):
    """开始新的训练会话"""
    session_id = str(uuid.uuid4())
    try:
        motion_analyzer = MotionAnalyzer(config.sport_type)
        motion_analyzer.start_session()

        if config.video_source == "webcam":
            rec_dir = Path("recordings")
            rec_dir.mkdir(exist_ok=True)
            video_config = VideoConfig(
                source_type=VideoSource.WEBCAM,
                enable_recording=config.enable_recording,
                recording_path=f"recordings/{session_id}.mp4" if config.enable_recording else None
            )
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported video source"})

        analyzer = VideoAnalyzer()
        if not analyzer.setup(video_config):
            raise HTTPException(status_code=500, detail="Failed to initialize video source")

        active_sessions[session_id] = analyzer
        motion_sessions[session_id] = {
            "analyzer": motion_analyzer,
            "start_time": time.time(),
            "sport_type": config.sport_type,
            "video_analyzer": analyzer
        }

        # 启动视频处理线程
        if hasattr(analyzer, 'video_processor') and analyzer.video_processor:
            analyzer.video_processor.start()

        return {"session_id": session_id, "status": "started", "config": config.dict()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/session/{session_id}/stop")
async def stop_session(session_id: str):
    """停止训练会话"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    analyzer = active_sessions[session_id]
    feedback = analyzer.stop_analysis()
    del active_sessions[session_id]

    if feedback:
        motion_frames = []
        if hasattr(analyzer, 'get_motion_frames'):
            motion_frames = analyzer.get_motion_frames()

        enhanced_feedback = {
            "overall_score": feedback.overall_score,
            "level": feedback.level.value,
            "strengths": feedback.strengths,
            "weaknesses": feedback.weaknesses,
            "suggestions": [{"title": s.title, "description": s.description, "drill": s.drill_recommendation} for s in
                            feedback.suggestions],
            "progress_notes": feedback.progress_notes,
            "motion_frames": len(motion_frames)
        }
        return enhanced_feedback

    return {"message": "Session stopped", "session_id": session_id}


@app.get("/api/v1/session/{session_id}/joint-data")
async def get_session_joint_data(session_id: str, summary: bool = True, limit: int = 0):
    """获取会话的关节数据"""
    if session_id not in motion_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        session_data = motion_sessions[session_id]
        motion_analyzer = session_data["analyzer"]
        if summary:
            joint_data = motion_analyzer.get_ai_ready_joint_data()
        else:
            joint_data = motion_analyzer.get_recent_joint_data(num_frames=limit if limit > 0 else 30)

        session_summary = motion_analyzer.get_session_summary()
        top_issues = motion_analyzer.get_top_issues(3)
        scores = [frame.score for frame in motion_analyzer.frame_history if hasattr(frame, 'score')]
        overall_score = np.mean(scores) * 100 if scores else 0

        response = {
            "session_id": session_id,
            "joint_data": joint_data,
            "summary": {
                "frame_count": session_summary.get("frame_count", 0),
                "duration": session_summary.get("duration", 0),
                "avg_score": round(overall_score, 1),
                "joint_count": len(joint_data)
            },
            "top_issues": top_issues,
            "strengths": _extract_strengths_from_joints(joint_data),
            "timestamp": time.time()
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取关节数据失败: {str(e)}")


def _extract_strengths_from_joints(joint_data: List[Dict]) -> List[str]:
    """从关节数据中提取优势"""
    strengths = []
    for joint in joint_data:
        deviation = joint.get("deviation", 999)
        if deviation == 0:
            strengths.append(f"{joint['joint_name']}角度控制完美")
        elif deviation < 5:
            strengths.append(f"{joint['joint_name']}角度控制精准")
        elif deviation < 10:
            strengths.append(f"{joint['joint_name']}角度控制良好")
    if not strengths: strengths.append("动作基础扎实")
    return strengths[:3]


@app.get("/api/v1/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """获取会话统计信息"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    analyzer = active_sessions[session_id]
    stats = analyzer.get_current_stats()
    return stats


@app.post("/api/v1/ai/analyze")
async def analyze_with_ai(request: Dict):
    """使用DeepSeek AI分析训练数据 (独立接口)"""
    try:
        session_id = request.get("session_id")
        user_level = request.get("user_level", "intermediate")
        joint_data = request.get("joint_data", [])

        # ... (保留原有的兜底逻辑) ...
        level_map = {"beginner": "初级", "intermediate": "中级", "advanced": "高级", "athlete": "运动员级"}
        user_level_zh = level_map.get(user_level, user_level)

        if not joint_data and session_id and session_id in motion_sessions:
            session_data = motion_sessions[session_id]
            motion_analyzer = session_data.get("analyzer")
            if motion_analyzer:
                joint_angle_data = motion_analyzer.get_joint_angle_data()
                for angle in joint_angle_data:
                    joint_data.append({
                        "joint_name": angle.get("joint_name", ""),
                        "angle": angle.get("angle", 0),
                        "deviation": angle.get("deviation", 0)
                    })

        if not joint_data:
            # 模拟数据兜底
            joint_data = [{"joint_name": "right_elbow", "angle": 120, "deviation": 5}]

        # 调用核心 AI 处理器
        processor = SiliconFlowAIProcessor() if SiliconFlowAIProcessor else None
        if processor:
            ai_feedback = processor.generate_sports_advice(joint_data, user_level_zh)
        else:
            ai_feedback = {"error": "AI Processor not initialized"}

        # ... (保留你原有的格式化逻辑) ...
        formatted_response = {
            "overall_score": ai_feedback.get("overall_score", 75),
            "level": ai_feedback.get("level", "良好"),
            "strengths": ai_feedback.get("strengths", []),
            "improvements": ai_feedback.get("improvements", []),
            "suggestions": ai_feedback.get("suggestions", []),
            "training_tips": ai_feedback.get("training_tips", []),
            "action_plan": ai_feedback.get("action_plan", {}),
            "dimensions": ai_feedback.get("dimensions", {}),
            "motivation": ai_feedback.get("motivation", "继续努力！"),
        }
        return formatted_response

    except Exception as e:
        print(f"❌ AI分析错误: {e}")
        return {"error": str(e)}


# ================= 核心修改：升级版上传接口 =================
# 在 api/main.py 中找到并替换 analyze_video 函数

@app.post("/api/v1/analysis/upload")
async def analyze_video(file: UploadFile = File(...)):
    """
    [终极修复版] 上传视频分析接口
    修复了 AI Processor 变量报错的问题，采用就地实例化策略。
    """
    print(f"📤 收到视频上传请求: {file.filename}")

    # 1. 保存文件
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    upload_id = str(uuid.uuid4())
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    upload_path = upload_dir / f"{upload_id}_{file.filename}"

    # 结果视频路径
    result_filename = f"analyzed_{upload_id}.mp4"
    result_path = Path("results") / result_filename

    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        print(f"✅ 文件保存成功: {upload_path}")

        # === 变量初始化 (防止 UnboundLocalError) ===
        base_analyzer = None
        basic_score = 0
        basic_feedback = None
        ai_analysis_reports = []

        # --- 第一部分：执行原有的基础分析 (MotionAnalyzer) ---
        try:
            video_config = VideoConfig(
                source_type=VideoSource.FILE,
                source_path=str(upload_path),
                fps=30,
                resolution=(1280, 720)
            )
            base_analyzer = VideoAnalyzer()
            if base_analyzer.setup(video_config):
                print("▶️ 开始基础视频分析...")
                basic_feedback = base_analyzer.analyze_video_sync()
                if basic_feedback:
                    basic_score = getattr(basic_feedback, 'overall_score', 0)
        except Exception as e:
            print(f"⚠️ 基础分析警告: {e}")

        # --- 第二部分：执行新的深度分析 (TennisSystem + Similarity + AI) ---
        if tennis_system:
            print("⚡ 开始深度网球系统分析 (YOLO + LSTM)...")
            try:
                # 1. 视觉处理
                deep_result = tennis_system.process_video(str(upload_path), str(result_path))

                if "shots" in deep_result:
                    for shot in deep_result["shots"]:
                        # 获取基础信息
                        shot_type = shot.get("type", "Unknown")
                        user_kps = shot.get("kps_seq", [])

                        print(f"  🎾 分析击球: {shot_type} (Conf: {shot.get('confidence', 0):.2f})")

                        # === A. 动作对比 (Similarity) ===
                        sim_score = 0
                        joint_metrics = []  # 详细关节数据

                        if comparator and user_kps:
                            try:
                                sim_score, feedback_text, joint_metrics = comparator.compare(user_kps, shot_type)
                                print(f"  🔍 {feedback_text}")
                            except Exception as ce:
                                print(f"  ⚠️ 对比分析出错: {ce}")

                        # === B. 调用 AI 生成建议 (AI Processor) ===
                        ai_advice = {}
                        if joint_metrics:  # 只有当有对比数据时才调 AI
                            try:
                                # [核心修复] 不要依赖全局变量，直接检查类是否存在
                                # 检查 SiliconFlowAIProcessor 是否是有效的类（不是占位符）
                                if SiliconFlowAIProcessor and getattr(SiliconFlowAIProcessor, '__name__',
                                                                      '') != 'PlaceholderClass':

                                    # 就地实例化 (Instance on the fly)
                                    # 这样完全避免了全局变量 ai_processor 未定义或 None 的问题
                                    processor = SiliconFlowAIProcessor()

                                    ai_advice = processor.generate_sports_advice(
                                        joint_data=joint_metrics,
                                        user_level="中级"
                                    )
                                else:
                                    print("⚠️ AI 模块未加载或不可用 (Placeholder)")

                            except Exception as ae:
                                print(f"  ⚠️ AI 生成过程出错: {ae}")
                                # 打印详细错误堆栈，方便调试
                                import traceback
                                traceback.print_exc()

                        # 存入结果
                        ai_analysis_reports.append({
                            "time": shot.get("frame", 0) / 30.0,
                            "type": shot_type,
                            "similarity_score": int(sim_score),
                            "ai_feedback": ai_advice
                        })
            except Exception as te:
                print(f"⚠️ 深度分析流程异常: {te}")
                import traceback
                traceback.print_exc()

        # --- 构造返回数据 ---
        # 安全获取基础分析的数据
        joint_data_basic = []
        if base_analyzer and hasattr(base_analyzer, 'get_joint_data_for_ai'):
            joint_data_basic = base_analyzer.get_joint_data_for_ai()

        basic_suggestions = []
        if basic_feedback and hasattr(basic_feedback, 'suggestions'):
            basic_suggestions = [
                {"title": getattr(s, 'title', ''), "description": getattr(s, 'description', '')}
                for s in basic_feedback.suggestions
            ]

        basic_level = '未知'
        if basic_feedback and hasattr(basic_feedback, 'level'):
            val = getattr(basic_feedback, 'level')
            basic_level = val.value if hasattr(val, 'value') else str(val)

        response_data = {
            "upload_id": upload_id,
            "filename": file.filename,
            "overall_score": basic_score if basic_score > 0 else 80,
            "level": basic_level,
            "video_url": f"http://localhost:8000/results/{result_filename}",

            "strengths": getattr(basic_feedback, 'strengths', []) if basic_feedback else [],
            "weaknesses": getattr(basic_feedback, 'weaknesses', []) if basic_feedback else [],
            "suggestions": basic_suggestions,

            "deep_analysis": ai_analysis_reports,
            "joint_data": joint_data_basic
        }

        return response_data

    except Exception as e:
        print(f"❌ API 顶层崩溃: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        try:
            if upload_path.exists():
                upload_path.unlink(missing_ok=True)
        except:
            pass

@app.post("/api/v1/analysis/frame")
async def analyze_frame(image: UploadFile = File(...)):
    """分析单帧图像 (保留)"""
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    analyzer = MotionAnalyzer("badminton")
    motion_frame = analyzer.process_frame(frame, 0.0)

    if motion_frame:
        return FrameAnalysis(
            timestamp=motion_frame.timestamp,
            score=motion_frame.score,
            issues=motion_frame.issues
        )
    raise HTTPException(status_code=500, detail="Failed to analyze frame")


# ================= 启动与关闭事件 =================

@app.on_event("startup")
async def startup_event():
    Path("uploads").mkdir(exist_ok=True)
    Path("recordings").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    print("Sports Analyzer API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    for session_id in list(active_sessions.keys()):
        analyzer = active_sessions[session_id]
        if hasattr(analyzer, 'stop_analysis'): analyzer.stop_analysis()
    active_sessions.clear()
    motion_sessions.clear()
    print("Sports Analyzer API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    # 统一使用 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)