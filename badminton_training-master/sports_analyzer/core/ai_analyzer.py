# core/ai_analyzer.py
"""
AI分析模块 - 使用 SiliconFlow API (Qwen/DeepSeek) 生成网球训练建议
适配多模态对比数据 (User vs Pro)
"""

import os
import logging
import numpy as np
import json
import requests
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIAnalysisRequest:
    """AI分析请求数据结构"""
    joint_angles_data: List[Dict]  # 关节角度数据
    user_level: str = "intermediate"  # 用户水平
    session_duration: float = 0.0  # 训练时长（秒）


class SiliconFlowAIProcessor:
    """
    基于 SiliconFlow API 的 AI 处理器
    核心功能：将关节数据转化为 Prompt，调用 LLM，解析返回的 JSON 建议
    """

    def __init__(self, api_key: str = None):
        """
        初始化 AI 处理器
        配置严格保留用户指定参数
        """
        # --- 用户指定配置 (绝对不改) ---
        self.api_key = api_key or "sk-rkvgiyahgxvgiphuqlxnwrjexucgdvrqlwwewdfpurjmtjgr"
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.model = "Qwen/Qwen2.5-7B-Instruct"  # 免费模型
        # -----------------------------

        # 请求头配置
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_sports_advice(self, joint_data: List[Dict], user_level: str = "中级") -> Dict[str, Any]:
        """
        [主入口] 生成网球训练建议

        Args:
            joint_data: 包含关节角度、偏差、职业参考值的列表
            user_level: 用户水平 (初级/中级/高级)

        Returns:
            Dict: 包含 overall_score, suggestions, action_plan 等完整 JSON
        """
        logger.info(f"🎾 [AI核心] 开始生成建议 | 用户: {user_level} | 数据点: {len(joint_data)}个")

        # 1. 基础数据校验
        if not joint_data:
            logger.warning("⚠️ 关节数据为空，返回默认响应")
            return self._get_default_response()

        # 2. 构建 Prompt (核心适配点)
        try:
            prompt = self._create_tennis_prompt(joint_data, user_level)
            logger.debug(f"📝 生成 Prompt 长度: {len(prompt)} 字符")
        except Exception as e:
            logger.error(f"❌ Prompt 构建失败: {e}", exc_info=True)
            return self._get_fallback_response(joint_data, user_level)

        # 3. 调用 API
        try:
            logger.info(f"📡 调用 API: {self.model} ...")
            ai_response_text = self._call_siliconflow_api(prompt)

            if ai_response_text:
                # 4. 解析结果
                parsed_data = self._parse_response(ai_response_text)

                # 5. 后处理与数据增强
                # 补充计算维度分 (用于前端雷达图)
                parsed_data["dimensions"] = self._calculate_dimensions(joint_data)

                # 补充激励语 (如果 AI 没返回)
                if "motivation" not in parsed_data or not parsed_data["motivation"]:
                    parsed_data["motivation"] = self._get_motivation(user_level)

                # 标记来源
                parsed_data["ai_source"] = "siliconflow_qwen"

                logger.info(f"✅ AI 分析成功 | 评分: {parsed_data.get('overall_score')}")
                return parsed_data
            else:
                logger.warning("⚠️ API 返回空内容")

        except Exception as e:
            logger.error(f"❌ AI 生成流程异常: {e}", exc_info=True)

        # 6. 兜底方案 (当 API 挂了或解析失败时)
        logger.info("🔄 启动本地规则引擎进行兜底...")
        return self._get_fallback_response(joint_data, user_level)

    def _call_siliconflow_api(self, prompt: str) -> Optional[str]:
        """
        执行 HTTP 请求调用 LLM API
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一名世界顶级的网球生物力学专家。请基于给出的【用户 vs 职业球员】骨骼对比数据，提供严厉且具体的纠正建议。请始终用中文回复，并严格按照要求的 JSON 格式输出，不要包含任何 Markdown 代码块标记。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": False,
            "response_format": {"type": "json_object"}  # 尝试强制 JSON 模式
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # 30秒超时
            )

            # 记录响应状态
            logger.debug(f"API 状态码: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"API 错误响应: {response.text}")
                return None

            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                logger.error(f"API 响应格式异常: {result}")
                return None

        except requests.exceptions.Timeout:
            logger.error("❌ API 请求超时")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("❌ 网络连接失败，请检查网络")
            return None
        except Exception as e:
            logger.error(f"❌ API 请求未知错误: {str(e)}")
            return None

    def _create_tennis_prompt(self, joint_data: List[Dict], user_level: str) -> str:
        """
        [核心] 构建包含【职业对比数据】的提示词

        利用 similarity.py 提供的 pro_angle, direction, is_key 等详细信息
        构建高上下文的 Prompt，确保 AI "看懂" 动作。
        """
        joint_lines = []
        problematic_joints = []
        excellent_joints = []

        # 统计数据
        total_joints = len(joint_data)
        perfect_count = 0

        # 遍历关节数据
        for j in joint_data:
            # 1. 基础信息
            name_en = j.get('joint_name', 'unknown')
            name_zh = self._translate_joint_name(name_en)
            user_ang = j.get('angle', 0)

            # 2. 职业对比信息 (适配 similarity.py)
            pro_ang = j.get('pro_angle')  # 职业参考值
            dev = j.get('deviation', 0)  # 偏差绝对值
            direction = j.get('direction', '')  # 方向: 过大/过小/过直
            status = j.get('status', '')  # 评价: 严重偏差/完美
            is_key = j.get('is_key', False)  # 是否核心发力点

            # 3. 构建单行描述
            # 格式: "- 右膝: 用户170.0° (职业130.0°) -> 偏差40.0° 过直 [严重偏差]"
            line = f"- {name_zh}: {float(user_ang):.1f}°"

            if pro_ang is not None:
                line += f" (职业{float(pro_ang):.1f}°)"

            if dev > 0:
                line += f" -> 偏差{float(dev):.1f}°"
                if direction:
                    line += f" {direction}"

            tags = []
            if status and status != "正常":
                tags.append(status)
            if is_key:
                tags.append("核心发力点")

            if tags:
                line += f" [{' | '.join(tags)}]"

            # 4. 筛选逻辑 (避免 Prompt 过长)
            # 只放入有偏差的、核心的、或者完美的关节
            if dev > 5 or is_key or status == "完美":
                joint_lines.append(line)

            # 5. 归类统计 (用于 Prompt 摘要)
            if dev > 12:
                problematic_joints.append(f"{name_zh}({direction})")
            elif dev < 5 and is_key:
                excellent_joints.append(name_zh)
                perfect_count += 1

        context_text = "\n".join(joint_lines)
        if not context_text:
            context_text = "所有监测关节数据均在标准范围内，与职业球员动作高度一致。"

        # 6. 组装最终 Prompt
        prompt = f"""
请分析一名{user_level}水平学员的网球动作。
以下数据由计算机视觉系统生成，对比了学员动作与职业球员(如费德勒)的标准模型。
(注：角度数据取自击球瞬间/发力关键帧，pro_angle 为职业球员同期角度)

【关节数据分析报告】
{context_text}

【摘要】
- 总分析关节：{total_joints}个
- 核心亮点（完美匹配）：{'、'.join(excellent_joints) if excellent_joints else '无明显亮点'}
- 严重问题（偏差过大）：{'、'.join(problematic_joints) if problematic_joints else '无严重问题'}

【分析任务】
请基于上述物理数据，输出一份严厉、专业、可执行的 JSON 诊断报告：

1. **overall_score**: 基于偏差值打分 (0-100)。偏差越大分数越低。
2. **level**: 评级 (职业级/优秀/良好/及格/需重练)。
3. **strengths**: 必须基于数据中【偏差小】且是【核心发力点】的关节生成。表扬其稳定性。
4. **improvements**: 必须基于数据中【偏差大】的问题生成。
   - 关键：必须结合 `direction` (过大/过小/过直) 来解释物理后果。
   - 例如：如果膝盖“过直”，请指出这会导致“重心高、无法借助地反力”。
5. **suggestions**: 针对上述问题，给出具体的训练 Drill（训练法）。
   - 例如：膝盖过直 -> 建议“背靠墙深蹲挥拍”。
   - 包含 title, description, priority, duration。
6. **action_plan**: 针对性的周训练计划。

【强制 JSON 格式】
请仅返回纯 JSON 字符串，不要包含 ```json 标记：
{{
  "overall_score": 0,
  "level": "string",
  "strengths": ["string"],
  "improvements": ["string"],
  "suggestions": [
    {{"title": "string", "description": "string", "priority": 1, "duration": "10min"}},
    {{"title": "string", "description": "string", "priority": 2, "duration": "15min"}}
  ],
  "training_tips": [
    {{"title": "热身", "description": "...", "icon": "thermometer", "priority": 1}}
  ],
  "action_plan": {{
    "focus": "string",
    "frequency": "string",
    "details": ["string"]
  }},
  "motivation": "string"
}}
"""
        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        清洗并解析 AI 返回的 JSON
        具有极强的容错性
        """
        try:
            logger.debug(f"🔍 解析原始响应: {response_text[:100]}...")

            if not response_text or response_text.strip() == "":
                return self._get_default_response()

            # 1. 移除 Markdown 代码块标记
            clean_text = re.sub(r'```json\s*', '', response_text)
            clean_text = re.sub(r'```\s*', '', clean_text).strip()

            # 2. 提取 JSON 对象 (寻找最外层的 {})
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, clean_text, re.DOTALL)

            if match:
                json_str = match.group()
                parsed = json.loads(json_str)

                # 3. 字段补全 (防止 AI 漏字段导致前端报错)
                required_fields = ["overall_score", "level", "strengths", "improvements", "suggestions"]
                for field in required_fields:
                    if field not in parsed:
                        logger.warning(f"⚠️ AI 响应缺失字段: {field}，使用默认值补全")
                        parsed[field] = self._get_default_value(field)

                return parsed
            else:
                logger.error("❌ 无法从响应中提取 JSON 对象")
                return self._get_default_response()

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 解码失败: {e}")
            # 尝试暴力修复（可选，这里先返回默认）
            return self._get_default_response()
        except Exception as e:
            logger.error(f"❌ 解析异常: {e}")
            return self._get_default_response()

    def _calculate_dimensions(self, joint_data: List[Dict]) -> Dict[str, float]:
        """
        计算各维度评分 (用于前端雷达图)
        基于实际偏差数据计算
        """
        if not joint_data:
            return {"angle": 0.75, "balance": 0.75, "fluidity": 0.70, "symmetry": 0.80}

        # 1. 角度控制 (Angle Control): 基于偏差均值
        deviations = [j.get("deviation", 0) for j in joint_data]
        avg_dev = np.mean(deviations) if deviations else 0
        # 偏差越小分越高 (0度->0.95, 20度->0.75, 50度->0.45)
        angle_score = max(0.4, 0.95 - (avg_dev / 100.0))

        # 2. 身体平衡 (Balance): 假设偏差大意味着平衡差
        # 如果有 hip/knee 的偏差，权重更大
        balance_penalty = 0
        for j in joint_data:
            name = j.get('joint_name', '')
            if 'knee' in name or 'hip' in name:
                balance_penalty += j.get('deviation', 0) * 0.005
        balance_score = max(0.4, angle_score - balance_penalty)

        # 3. 动作流畅 (Fluidity): 模拟值，通常与角度控制正相关
        fluidity_score = max(0.4, angle_score * 0.9)

        # 4. 对称性 (Symmetry): 如果有左右关节数据，计算差值
        symmetry_score = 0.8  # 默认
        # (此处可扩展左右关节对比逻辑)

        return {
            "angle": round(angle_score, 2),
            "balance": round(balance_score, 2),
            "fluidity": round(fluidity_score, 2),
            "symmetry": round(symmetry_score, 2)
        }

    def _translate_joint_name(self, joint_name: str) -> str:
        """关节名称中文化"""
        translations = {
            "right_elbow": "右肘", "left_elbow": "左肘",
            "right_shoulder": "右肩", "left_shoulder": "左肩",
            "right_hip": "右髋", "left_hip": "左髋",
            "right_knee": "右膝", "left_knee": "左膝",
            "right_ankle": "右踝", "left_ankle": "左踝",
            "right_wrist": "右手腕", "left_wrist": "左手腕"
        }
        return translations.get(joint_name, joint_name)

    def _get_motivation(self, user_level: str) -> str:
        """根据水平生成激励语"""
        motivations = {
            "初级": "万丈高楼平地起，专注于每一次挥拍！",
            "中级": "你已经具备了良好的框架，现在的目标是极致的精准！",
            "高级": "像职业球员一样思考，细节决定成败！",
            "运动员级": "追求卓越，突破生理极限！"
        }
        return motivations.get(user_level, "保持热爱，奔赴山海！")

    def _get_default_value(self, field: str):
        defaults = {
            "overall_score": 75,
            "level": "良好",
            "strengths": ["动作识别完成"],
            "improvements": ["正在计算具体数据"],
            "suggestions": [{"title": "保持训练", "description": "请继续保持当前状态", "priority": 1}]
        }
        return defaults.get(field, None)

    def _get_default_response(self) -> Dict[str, Any]:
        """完全空的默认响应"""
        return {
            "overall_score": 70,
            "level": "分析中",
            "strengths": ["系统就绪"],
            "improvements": ["等待数据输入"],
            "suggestions": [{"title": "准备", "description": "请开始您的动作", "priority": 1}],
            "action_plan": {"focus": "基础", "details": ["热身"]},
            "dimensions": {"angle": 0.7, "balance": 0.7, "fluidity": 0.7, "symmetry": 0.7},
            "motivation": "准备开始..."
        }

    def _get_fallback_response(self, joint_data: List[Dict], user_level: str) -> Dict[str, Any]:
        """
        备用规则引擎
        当 API 调用失败或超时时，使用本地规则生成“看起来像真的一样”的建议
        """
        logger.warning("⚠️ 切换至本地规则引擎生成建议")

        deviations = [j.get("deviation", 0) for j in joint_data]
        avg_dev = np.mean(deviations) if deviations else 15

        # 计算分数
        score = max(50, 100 - int(avg_dev * 1.5))

        # 评级
        if score >= 90:
            level = "优秀"
        elif score >= 80:
            level = "良好"
        elif score >= 60:
            level = "及格"
        else:
            level = "需改进"

        # 生成改进项
        improvements = []
        for j in joint_data:
            dev = j.get('deviation', 0)
            if dev > 15:
                n = self._translate_joint_name(j.get('joint_name', ''))
                d = j.get('direction', '偏差')
                improvements.append(f"{n} {d} (偏差{dev:.1f}°)")

        if not improvements:
            improvements = ["整体动作标准，建议增加击球力量训练"]

        # 生成优势
        strengths = []
        for j in joint_data:
            if j.get('deviation', 99) < 8:
                n = self._translate_joint_name(j.get('joint_name', ''))
                strengths.append(f"{n} 控制稳定")
        if not strengths: strengths = ["完成整套动作"]

        return {
            "overall_score": score,
            "level": level,
            "strengths": strengths[:3],  # 取前3个
            "improvements": improvements[:3],
            "suggestions": [
                {"title": "基础巩固", "description": "由于网络原因无法获取 AI 深度建议，请重点练习基础挥拍。",
                 "priority": 1, "duration": "10min"},
                {"title": "多球练习", "description": "重复定点击球，通过大量重复固定肌肉记忆。", "priority": 2,
                 "duration": "20min"}
            ],
            "action_plan": {
                "focus": "稳定性训练",
                "frequency": "每周3次",
                "details": ["周一：正手定点", "周三：反手定点", "周五：发球练习"]
            },
            "dimensions": self._calculate_dimensions(joint_data),
            "motivation": "网络连接不稳定，已切换至本地规则模式。训练不能停！",
            "ai_source": "fallback_rule_engine"
        }


class AISuggestionSystem:
    """
    AI建议系统包装类
    兼容旧代码调用接口，底层代理给 SiliconFlowAIProcessor
    """

    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        # 实例化新处理器
        self.ai_processor = SiliconFlowAIProcessor() if use_ai else None

    def generate_ai_feedback(self, user_level: str, motion_frames: List,
                             historical_feedback: List) -> Dict[str, Any]:
        """
        兼容旧接口：从 MotionFrame 列表提取数据并调用 AI
        """
        joint_data = []

        # 尝试提取最后一帧的数据作为代表
        if motion_frames and len(motion_frames) > 0:
            latest = motion_frames[-1]
            if hasattr(latest, 'joint_angles'):
                for angle in latest.joint_angles:
                    # 构造兼容格式
                    joint_data.append({
                        "joint_name": getattr(angle, 'joint_name', 'unknown'),
                        "angle": getattr(angle, 'angle', 0),
                        "deviation": getattr(angle, 'deviation', 0),
                        # 注意：这里可能缺少 pro_angle，_create_tennis_prompt 会处理这种情况
                        "status": "需改进" if getattr(angle, 'deviation', 0) > 10 else "良好"
                    })

        if self.use_ai and self.ai_processor:
            return self.ai_processor.generate_sports_advice(joint_data, user_level)
        else:
            if self.ai_processor:
                return self.ai_processor._get_fallback_response(joint_data, user_level)
            return {}


# 便捷函数 (Module Level Helper)
def get_ai_suggestions(joint_data: List[Dict], **kwargs) -> Dict[str, Any]:
    """获取AI建议的便捷函数"""
    processor = SiliconFlowAIProcessor()
    return processor.generate_sports_advice(
        joint_data,
        user_level=kwargs.get("user_level", "中级")
    )


# 测试代码
if __name__ == "__main__":
    # 模拟 similarity.py 输出的高级数据
    test_data = [
        {
            "joint_name": "right_knee",
            "angle": 170.0,
            "pro_angle": 130.0,
            "deviation": 40.0,
            "direction": "过直",
            "status": "严重偏差",
            "is_key": True
        },
        {
            "joint_name": "right_elbow",
            "angle": 90.0,
            "pro_angle": 95.0,
            "deviation": 5.0,
            "direction": "稍小",
            "status": "良好",
            "is_key": False
        }
    ]

    print("🚀 正在测试 AI 生成 (使用 Qwen)...")

    # 实例化 (API Key 已内置)
    processor = SiliconFlowAIProcessor()
    result = processor.generate_sports_advice(test_data, "中级")

    print("\n✅ AI反馈结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))