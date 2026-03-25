# simple_main.py - 简化版API，用于测试前端连接
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime
import asyncio
import json

app = FastAPI(title="Sports Analyzer API - 简化版")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储模拟数据
active_sessions = {}
websocket_connections = []


class SessionConfig(BaseModel):
    sport_type: str = "badminton"
    video_source: str = "webcam"
    enable_recording: bool = False
    user_id: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Sports Analyzer API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/sports")
async def get_sports():
    return {
        "sports": [
            {"id": "badminton", "name": "羽毛球"},
            {"id": "tennis", "name": "网球"},
            {"id": "basketball", "name": "篮球"},
            {"id": "golf", "name": "高尔夫"},
            {"id": "yoga", "name": "瑜伽"},
            {"id": "running", "name": "跑步"}
        ]
    }


@app.post("/api/v1/session/start")
async def start_session(config: SessionConfig):
    """开始训练会话（模拟）"""
    session_id = str(uuid.uuid4())

    active_sessions[session_id] = {
        "id": session_id,
        "start_time": datetime.now(),
        "config": config.dict(),
        "status": "active",
        "frame_count": 0,
        "scores": []
    }

    print(f"✅ 会话开始: {session_id}")
    print(f"   运动类型: {config.sport_type}")
    print(f"   视频源: {config.video_source}")

    return {
        "session_id": session_id,
        "status": "started",
        "config": config.dict()
    }


@app.post("/api/v1/session/{session_id}/stop")
async def stop_session(session_id: str):
    """停止训练会话（模拟）"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions.pop(session_id)
    duration = (datetime.now() - session["start_time"]).total_seconds()

    # 模拟计算平均分数
    avg_score = 85.5 if session["scores"] else 0
    if session["scores"]:
        avg_score = sum(session["scores"]) / len(session["scores"])

    print(f"✅ 会话停止: {session_id}")
    print(f"   时长: {duration:.1f}秒")
    print(f"   平均分: {avg_score:.1f}")

    return {
        "session_id": session_id,
        "status": "stopped",
        "duration": duration,
        "overall_score": round(avg_score, 1),
        "level": "优秀" if avg_score >= 90 else "良好" if avg_score >= 70 else "需要改进",
        "strengths": ["动作流畅", "姿势标准", "节奏稳定"],
        "weaknesses": ["挥拍力度不足", "脚步移动稍慢"],
        "suggestions": [
            {
                "title": "加强核心力量训练",
                "description": "建议每天做15分钟平板支撑",
                "drill": "3组×30秒平板支撑"
            },
            {
                "title": "提高挥拍速度",
                "description": "使用较轻的训练拍进行速度练习",
                "drill": "100次快速挥拍练习"
            }
        ],
        "progress_notes": "整体表现良好，继续坚持训练"
    }


@app.get("/api/v1/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """获取会话统计（模拟实时数据）"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]
    duration = (datetime.now() - session["start_time"]).total_seconds()

    # 模拟实时数据
    import random
    session["frame_count"] += random.randint(1, 5)
    current_score = 80 + random.uniform(-5, 10)
    session["scores"].append(current_score)

    stats = {
        "session_id": session_id,
        "active": True,
        "duration": round(duration, 2),
        "frame_count": session["frame_count"],
        "current_score": round(current_score, 1),
        "actual_fps": 25.0 + random.uniform(-2, 2),
        "status": "analyzing"
    }

    print(f"📊 会话统计: {session_id}")
    print(f"   帧数: {stats['frame_count']}, 分数: {stats['current_score']}, FPS: {stats['actual_fps']:.1f}")

    return stats


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket端点（模拟实时数据流）"""
    print(f"🌐 WebSocket连接尝试: {session_id}")

    await websocket.accept()
    websocket_connections.append(websocket)

    print(f"✅ WebSocket连接建立: {session_id}")

    try:
        while True:
            # 模拟发送实时数据
            if session_id in active_sessions:
                # 直接生成数据，避免调用其他端点
                import random
                session = active_sessions[session_id]
                duration = (datetime.now() - session["start_time"]).total_seconds()
                session["frame_count"] += random.randint(1, 5)
                current_score = 80 + random.uniform(-5, 10)

                stats = {
                    "session_id": session_id,
                    "active": True,
                    "duration": round(duration, 2),
                    "frame_count": session["frame_count"],
                    "current_score": round(current_score, 1),
                    "actual_fps": 25.0 + random.uniform(-2, 2),
                    "status": "analyzing"
                }

                await websocket.send_json({
                    "type": "stats",
                    "data": stats
                })

                print(f"📤 WebSocket发送数据到 {session_id}: 分数={stats['current_score']}")

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Session not found"
                })
                print(f"❌ WebSocket错误: 会话 {session_id} 不存在")
                break

            await asyncio.sleep(1)  # 每秒发送一次

    except Exception as e:
        print(f"❌ WebSocket错误: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        await websocket.close()
        print(f"🔌 WebSocket连接关闭: {session_id}")


@app.post("/api/v1/analysis/upload")
async def analyze_video(file: UploadFile = File(...)):
    """分析上传的视频（模拟）"""
    print(f"📤 收到视频上传: {file.filename}")

    # 模拟处理时间
    await asyncio.sleep(2)

    print("✅ 视频分析完成")

    return {
        "message": "视频分析完成",
        "overall_score": 82.5,
        "level": "良好",
        "strengths": ["动作标准", "节奏感好", "反应迅速"],
        "weaknesses": ["力量不足", "持久力需提高"],
        "suggestions": [
            {
                "title": "反应训练",
                "description": "进行快速反应训练",
                "drill": "使用反应球练习，每天10分钟"
            },
            {
                "title": "力量训练",
                "description": "加强上肢和核心力量",
                "drill": "每周3次力量训练，每次30分钟"
            }
        ],
        "progress_notes": "视频分析显示基本动作正确，建议加强力量和耐力训练"
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🤖 AI运动分析系统 - 简化版API")
    print("=" * 60)
    print("API地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("支持的运动: http://localhost:8000/api/v1/sports")
    print("=" * 60)
    print("💡 测试步骤:")
    print("1. 前端访问: http://localhost:8080 (需要启动前端HTTP服务器)")
    print("2. 点击'开始训练'按钮")
    print("3. 观察控制台输出")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)