# core/llm_processor.py
"""
本地大语言模型处理器 - 用于调用Qwen模型
"""

import json
import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

logger = logging.getLogger(__name__)


class LocalLLMProcessor:
    """本地大语言模型处理器"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = None):
        """
        初始化本地LLM处理器

        Args:
            model_path: 模型路径或HuggingFace模型名称
            device: 设备 (cuda/cpu/mps)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        logger.info(f"初始化LLM处理器，设备: {self.device}")

    def load_model(self) -> bool:
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            self.is_loaded = True

            logger.info("模型加载成功")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.is_loaded = False
            return False

    def generate_response(self, prompt: str,
                          max_new_tokens: int = 1024,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> str:
        """
        生成AI响应

        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: 核采样参数

        Returns:
            AI生成的文本
        """
        if not self.is_loaded:
            if not self.load_model():
                return "模型加载失败，无法生成建议"

        try:
            # 准备输入
            messages = [
                {"role": "system", "content": "你是一名专业的体育教练AI助手，专门分析运动姿势并提供训练建议。"},
                {"role": "user", "content": prompt}
            ]

            # 使用tokenizer的apply_chat_template方法
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt")
            else:
                # 备用方法
                text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                inputs = self.tokenizer(text, return_tensors="pt")

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码响应
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取助手回复
            if "assistant:" in response.lower():
                response = response.split("assistant:", 1)[-1].strip()

            logger.info(f"生成响应成功，长度: {len(response)}")
            return response

        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            return f"AI生成失败: {str(e)}"

    def generate_sports_advice(self, joint_data: List[Dict],
                               sport_type: str = "羽毛球",
                               user_level: str = "中级") -> Dict[str, Any]:
        """
        生成运动建议

        Args:
            joint_data: 关节数据列表
            sport_type: 运动类型
            user_level: 用户水平

        Returns:
            结构化建议
        """
        # 构建提示词
        prompt = self._create_sports_prompt(joint_data, sport_type, user_level)

        # 生成AI响应
        ai_response = self.generate_response(prompt)

        # 解析响应为结构化数据
        return self._parse_ai_response(ai_response, sport_type, user_level)

    def _create_sports_prompt(self, joint_data: List[Dict],
                              sport_type: str, user_level: str) -> str:
        """创建运动分析提示词"""

        # 格式化关节数据
        joint_info = []
        for joint in joint_data:
            joint_info.append(
                f"- {joint['joint_name']}: 当前角度 {joint['angle']:.1f}°, "
                f"最优范围 {joint['optimal_range'][0]}-{joint['optimal_range'][1]}°, "
                f"偏差 {joint['deviation']:.1f}°"
            )

        joint_text = "\n".join(joint_info)

        prompt = f"""作为一名专业的{sport_type}教练，请分析以下运动员的关节数据，并提供详细的训练建议。

运动员水平：{user_level}
运动类型：{sport_type}

关节数据分析：
{joint_text}

请按照以下JSON格式提供分析结果：
{{
  "overall_score": 0-100的整数分数,
  "level": "优秀/良好/合格/需改进",
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["需要改进的地方1", "需要改进的地方2"],
  "technical_suggestions": [
    {{
      "title": "技术建议标题",
      "description": "具体建议描述",
      "priority": 1-3的数字
    }}
  ],
  "training_exercises": [
    {{
      "name": "练习名称",
      "description": "练习描述",
      "sets": "组数",
      "reps": "次数",
      "frequency": "每周频率"
    }}
  ],
  "key_points": ["关键要点1", "关键要点2", "关键要点3"],
  "motivation": "鼓励的话语"
}}

请确保：
1. 分数基于关节偏差和运动技术要求合理评估
2. 建议具体、可操作
3. 针对{sport_type}的特点提供专业建议
4. 考虑{user_level}水平的训练需求

请直接输出JSON格式的分析结果，不要添加其他文字。"""

        return prompt

    def _parse_ai_response(self, response: str, sport_type: str,
                           user_level: str) -> Dict[str, Any]:
        """解析AI响应"""

        # 尝试从响应中提取JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            try:
                json_str = response[json_start:json_end]
                parsed_data = json.loads(json_str)

                # 确保所有必要字段都存在
                defaults = {
                    "overall_score": 75,
                    "level": "良好",
                    "strengths": ["动作基础扎实"],
                    "weaknesses": ["细节需要优化"],
                    "technical_suggestions": [
                        {
                            "title": "关节控制练习",
                            "description": "加强关键关节的角度控制",
                            "priority": 2
                        }
                    ],
                    "training_exercises": [
                        {
                            "name": "基础动作练习",
                            "description": "重复标准动作，形成肌肉记忆",
                            "sets": "3",
                            "reps": "10-15",
                            "frequency": "每周3次"
                        }
                    ],
                    "key_points": ["保持身体稳定", "注意动作节奏", "呼吸配合"],
                    "motivation": "坚持训练，每天进步一点点！"
                }

                # 合并默认值和AI生成的数据
                for key, default_value in defaults.items():
                    if key not in parsed_data or not parsed_data[key]:
                        parsed_data[key] = default_value

                return parsed_data

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")

        # 如果解析失败，返回默认响应
        return {
            "overall_score": 75,
            "level": "良好",
            "strengths": ["使用AI分析能力", "动作识别准确"],
            "weaknesses": ["具体分析需要更多数据"],
            "technical_suggestions": [
                {
                    "title": "AI辅助训练",
                    "description": response[:200] if response else "AI正在分析您的动作数据",
                    "priority": 1
                }
            ],
            "training_exercises": [
                {
                    "name": "常规训练",
                    "description": "保持常规训练计划",
                    "sets": "3",
                    "reps": "12",
                    "frequency": "每周3-4次"
                }
            ],
            "key_points": ["坚持训练", "关注动作细节", "定期评估进步"],
            "motivation": "AI教练与你一起进步！"
        }

    def unload_model(self):
        """卸载模型以释放内存"""
        if self.model:
            del self.model
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("模型已卸载")


# 单例实例
_llm_processor = None


def get_llm_processor(model_path: str = None) -> LocalLLMProcessor:
    """获取LLM处理器实例（单例模式）"""
    global _llm_processor

    if _llm_processor is None:
        # 设置默认模型路径
        default_paths = [
            "Qwen/Qwen2.5-7B-Instruct",  # HuggingFace模型名称
            "./models/qwen",  # 本地模型路径1
            "./qwen_model",  # 本地模型路径2
            "C:/models/qwen"  # Windows本地路径
        ]

        # 如果用户提供了路径，优先使用
        if model_path:
            _llm_processor = LocalLLMProcessor(model_path)
        else:
            # 尝试默认路径
            for path in default_paths:
                try:
                    _llm_processor = LocalLLMProcessor(path)
                    break
                except:
                    continue

            if _llm_processor is None:
                logger.warning("未找到Qwen模型，将使用规则引擎")
                _llm_processor = LocalLLMProcessor("dummy")  # 创建虚拟实例

    return _llm_processor