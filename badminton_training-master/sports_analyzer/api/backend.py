from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import asyncio
import uuid
import json
from typing import Dict, List
from datetime import datetime
import os
import cv2
import time
import numpy as np
import os

from sports_analyzer.core.ai_analyzer import AISuggestionSystem
from sports_analyzer.core.video_processor import VideoProcessor, VideoConfig, VideoSource
from sports_analyzer.core.motion_analyzer import MotionAnalyzer
from sports_analyzer.core.feedback_system import FeedbackSystem

ai_suggestion_system = AISuggestionSystem()
app = FastAPI(title="运动动作分析系统 API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342",
        "http://127.0.0.1:63342"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储会话和状态
sessions: Dict[str, dict] = {}
active_connections: Dict[str, WebSocket] = {}

# 每个会话的训练模型实例
session_models: Dict[str, dict] = {}


@app.post('/api/get_ai_feedback')
async def get_ai_feedback(request: Request):
    """获取AI生成的个性化反馈"""
    try:
        data = await request.json()

        # 获取用户水平
        user_level = data.get('user_level', 'intermediate')

        # 获取动作数据（这里需要根据你的实际数据结构调整）
        motion_frames = []  # 这里应该从数据库或内存中获取

        # 获取历史反馈
        historical_feedback = []  # 这里应该从数据库中获取用户历史

        # 生成AI反馈
        ai_feedback = ai_suggestion_system.generate_ai_feedback(
            user_level=user_level,
            motion_frames=motion_frames,
            historical_feedback=historical_feedback
        )

        return {
            'success': True,
            'ai_feedback': ai_feedback
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )



@app.post('/api/set_user_level')
async def set_user_level(request: Request):
    """设置用户水平"""
    try:
        data = await request.json()
        user_level = data.get('level', 'intermediate')

        # 这里应该保存到用户配置中
        # user_config.set_level(user_level)

        return {
            'success': True,
            'message': f'用户水平已设置为: {user_level}'
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )

# 运动类型映射


def initialize_session_models(session_id: str, sport_type: str):
    """为会话初始化训练模型"""
    try:
        # 初始化各个模型
        motion_analyzer = MotionAnalyzer()
        feedback_system = FeedbackSystem()

        # 注意：这里需要提供一个有效的视频文件路径
        # 因为你使用的是 WEBCAM 类型，但实际不需要摄像头
        # 修改配置为使用文件类型
        video_config = VideoConfig(
            source_type=VideoSource.FILE,  # 改为 FILE
            source_path=None,  # 需要提供实际的文件路径
            fps=30,
            resolution=(1280, 720),
            buffer_size=10,
            enable_recording=False
        )

        video_processor = VideoProcessor(video_config)

        # 添加动作分析处理器
        def analyze_frame_callback(frame_data):
            # 使用动作分析器处理帧
            motion_frame = motion_analyzer.process_frame(
                frame_data.original_frame,
                frame_data.timestamp
            )

            if motion_frame:
                # 绘制骨架
                annotated_frame = motion_analyzer.draw_skeleton(
                    frame_data.original_frame,
                    motion_frame
                )
                frame_data.processed_frame = annotated_frame

                # 保存分析结果
                frame_data.metadata = {
                    "score": motion_frame.score,
                    "issues": motion_frame.issues,
                    "joint_angles": [
                        {
                            "joint": angle.joint_name,
                            "angle": angle.angle,
                            "deviation": angle.deviation
                        }
                        for angle in motion_frame.joint_angles
                    ]
                }

            return frame_data

        video_processor.add_processor(analyze_frame_callback)

        # 存储模型实例
        session_models[session_id] = {
            "motion_analyzer": motion_analyzer,
            "feedback_system": feedback_system,
            "video_processor": video_processor,
            "analysis_results": [],
            "is_processing": False
        }

        # 初始化视频处理器
        success = video_processor.initialize()
        return success

    except Exception as e:
        print(f"初始化模型失败: {str(e)}")
        return False


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "运动动作分析系统 API",
        "version": "1.0.0",
        "endpoints": {
            "start_session": "POST /api/v1/session/start",
            "stop_session": "POST /api/v1/session/{session_id}/stop",
            "upload_video": "POST /api/v1/analysis/upload",
            "websocket": "WS /ws/{session_id}"
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/session/start")
async def start_session(data: dict):
    """开始一个新的训练会话"""
    try:
        session_id = str(uuid.uuid4())
        video_source = data.get("video_source", "webcam")

        # 初始化训练模型
        success = initialize_session_models(session_id)
        if not success:
            return JSONResponse(
                status_code=500,
                content={"error": "训练模型初始化失败"}
            )

        # 启动视频处理
        models = session_models[session_id]
        models["video_processor"].start()
        models["is_processing"] = True

        # 存储会话信息
        sessions[session_id] = {
            "session_id": session_id,
            "video_source": video_source,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "frame_count": 0,
            "data_points": []
        }

        # 启动实时数据发送任务
        asyncio.create_task(send_realtime_data(session_id))

        return {
            "session_id": session_id,
            "message": "训练会话已开始",
            "start_time": sessions[session_id]["start_time"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"启动会话失败: {str(e)}"}
        )


async def send_realtime_data(session_id: str):
    """实时发送数据到WebSocket客户端"""
    if session_id not in session_models:
        return

    while session_id in sessions and sessions[session_id]["status"] == "active":
        try:
            if session_id in active_connections:
                websocket = active_connections[session_id]

                # 获取当前统计数据
                stats = get_current_stats(session_id)

                # 获取实时帧（如果有）
                models = session_models[session_id]
                video_processor = models["video_processor"]

                frame_data = video_processor.get_processed_frame()
                if frame_data and frame_data.metadata:
                    stats.update(frame_data.metadata)
                    models["analysis_results"].append(frame_data.metadata)

                    # 限制结果缓存大小
                    if len(models["analysis_results"]) > 300:
                        models["analysis_results"].pop(0)

                # 发送数据
                realtime_data = {
                    "type": "stats",
                    "data": stats,
                    "timestamp": datetime.now().isoformat()
                }

                await websocket.send_json(realtime_data)

                # 更新会话统计
                sessions[session_id]["frame_count"] = stats.get("frame_count", 0)

        except Exception as e:
            print(f"发送实时数据失败: {e}")
            break

        await asyncio.sleep(0.1)  # 10 FPS


def get_current_stats(session_id: str) -> dict:
    """获取当前统计信息"""
    if session_id not in session_models:
        return {}

    models = session_models[session_id]
    video_processor = models["video_processor"]

    stats = video_processor.get_stats()

    # 添加运动分析相关的统计
    if models["analysis_results"]:
        recent_scores = [r.get("score", 0) for r in models["analysis_results"][-30:]]
        if recent_scores:
            stats["current_score"] = np.mean(recent_scores)
            stats["avg_score"] = np.mean(recent_scores)
            stats["best_score"] = np.max(recent_scores)

    return stats


@app.post("/api/v1/session/{session_id}/stop")
async def stop_session(session_id: str):
    """停止训练会话并返回分析结果"""
    if session_id not in sessions:
        return JSONResponse(
            status_code=404,
            content={"error": "会话不存在"}
        )

    # 停止视频处理
    if session_id in session_models:
        models = session_models[session_id]
        models["video_processor"].stop()
        models["is_processing"] = False

        # 生成反馈报告
        if models["analysis_results"]:
            # 这里需要将分析结果转换为 MotionFrame 格式
            # 简单示例：使用平均分数生成反馈
            scores = [r.get("score", 0) for r in models["analysis_results"] if "score" in r]
            avg_score = np.mean(scores) if scores else 0.0

            feedback = models["feedback_system"].generate_feedback(
                models["analysis_results"]  # 这里需要转换为 MotionFrame 列表
            )

            # 清理模型
            del session_models[session_id]

            # 更新会话状态
            session = sessions[session_id]
            session["status"] = "completed"
            session["end_time"] = datetime.now().isoformat()

            return {
                "session_id": session_id,
                "session_info": {
                    "sport_type": session["sport_type"],
                    "duration": "45分钟",
                    "total_frames": session["frame_count"]
                },
                "overall_score": avg_score * 100,
                "strengths": ["动作分析完成"],
                "weaknesses": ["请查看详细报告"],
                "suggestions": [
                    {
                        "title": "继续训练",
                        "description": "建议保持定期训练",
                        "drill": "每周3-5次训练"
                    }
                ]
            }

    return JSONResponse(
        status_code=500,
        content={"error": "分析结果生成失败"}
    )


# backend.py 中的 upload_video 函数修改：
@app.post("/api/v1/analysis/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件进行分析"""
    try:
        print(f"=== 开始视频上传分析 ===")
        print(f"文件名: {file.filename}")

        # 创建临时文件
        file_id = str(uuid.uuid4())
        original_filename = file.filename
        temp_file = f"uploads/temp_{file_id}_{original_filename}"

        # 确保上传目录存在
        os.makedirs("uploads", exist_ok=True)

        print(f"保存到: {temp_file}")

        # 保存上传的文件
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = os.path.getsize(temp_file)
        print(f"文件大小: {file_size / 1024:.1f} KB")

        # 初始化视频处理器（文件模式）
        video_config = VideoConfig(
            source_type=VideoSource.FILE,
            source_path=temp_file,
            fps=30,
            resolution=(1280, 720)
        )

        video_processor = VideoProcessor(video_config)
        motion_analyzer = MotionAnalyzer()
        motion_analyzer.start_session()

        print(f"视频处理器初始化...")

        # 初始化视频源
        if not video_processor.initialize():
            print("❌ 视频处理器初始化失败")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return JSONResponse(
                status_code=500,
                content={"error": "无法初始化视频源"}
            )

        print("✅ 视频处理器初始化成功")

        # 添加分析处理器
        analysis_results = []
        frame_scores = []

        def analyze_frame_callback(frame_data):
            motion_frame = motion_analyzer.process_frame(
                frame_data.original_frame,
                frame_data.timestamp
            )

            if motion_frame:
                print(f"✓ 分析第 {frame_data.frame_id} 帧, 评分: {motion_frame.score:.1f}")

                frame_data.metadata = {
                    "score": motion_frame.score,
                    "issues": motion_frame.issues
                }
                analysis_results.append(motion_frame)
                frame_scores.append(motion_frame.score)

            return frame_data

        video_processor.add_processor(analyze_frame_callback)

        print("开始视频处理...")

        # 初始化并处理视频
        video_processor.start()

        # 等待处理完成 - 使用更智能的方式
        max_wait_time = 300  # 5分钟超时
        check_interval = 0.5
        last_frame_count = 0
        no_progress_count = 0

        print("等待视频处理...")

        for i in range(int(max_wait_time / check_interval)):
            await asyncio.sleep(check_interval)

            current_frame_count = video_processor.stats["processed_frames"]

            if current_frame_count > last_frame_count:
                print(f"进度: 已处理 {current_frame_count} 帧, FPS: {video_processor.stats['fps']:.1f}")
                last_frame_count = current_frame_count
                no_progress_count = 0
            else:
                no_progress_count += 1

            # 如果连续5次检查都没有进展，认为处理完成
            if no_progress_count >= 5:
                print(f"视频处理似乎已完成 (连续 {no_progress_count} 次无进展)")
                break

            # 如果视频处理器已停止，也结束等待
            if not video_processor.is_running:
                print("视频处理器已停止")
                break

        print("停止视频处理器...")
        video_processor.stop()

        final_frame_count = video_processor.stats["processed_frames"]
        print(f"处理完成，总共处理 {final_frame_count} 帧")

        # 生成反馈
        if analysis_results:
            print(f"分析结果数量: {len(analysis_results)}")

            # 获取当前分数信息
            current_scores = motion_analyzer.get_current_scores()
            session_stats = motion_analyzer.get_session_statistics()

            print(f"当前分数: {current_scores}")
            print(f"会话统计: {session_stats}")

            # 计算平均分
            avg_score = np.mean(frame_scores) * 100 if frame_scores else 0

            feedback = {
                "filename": original_filename,
                "message": "视频分析完成",
                "frame_count": final_frame_count,
                "fps": video_processor.stats["fps"],
                "overall_score": round(avg_score, 1),
                "angle_control": current_scores.get("angle_control", 0),
                "balance": current_scores.get("balance", 0),
                "fluidity": current_scores.get("fluidity", 0),
                "symmetry": current_scores.get("symmetry", 0),
                "strengths": ["视频分析成功完成"],
                "weaknesses": current_scores.get("issues", ["请查看详细分析报告"]),
                "suggestions": [
                    {
                        "title": "继续训练",
                        "description": "建议保持定期训练",
                        "drill": "每周3-5次训练"
                    }
                ],
                "session_stats": session_stats,
                "debug_info": {
                    "processed_frames": final_frame_count,
                    "analysis_frames": len(analysis_results),
                    "avg_frame_score": round(avg_score, 1)
                }
            }

            # 清理临时文件
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")
            except Exception as e:
                print(f"清理文件时出错: {e}")

            print("✅ 视频分析完成，返回结果")
            return feedback
        else:
            print("❌ 没有分析结果")

            # 清理临时文件
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

            return JSONResponse(
                status_code=500,
                content={
                    "error": "视频分析失败，未获取到有效数据",
                    "debug_info": {
                        "processed_frames": video_processor.stats["processed_frames"],
                        "is_running": video_processor.is_running
                    }
                }
            )

    except Exception as e:
        print(f"❌ 视频上传分析出错: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"视频上传失败: {str(e)}"}
        )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket 连接，用于实时数据传输"""
    await websocket.accept()

    if session_id not in sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    # 存储连接
    active_connections[session_id] = websocket

    try:
        while True:
            # 等待消息或连接关闭
            data = await websocket.receive_text()
            # 可以处理来自客户端的控制消息
            # 例如：暂停、调整参数等

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in active_connections:
            del active_connections[session_id]


@app.get("/api/v1/sessions")
async def list_sessions():
    """获取所有活动会话（调试用）"""
    return {
        "active_sessions": len(active_connections),
        "total_sessions": len(sessions),
        "sessions": sessions
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )