"""
OpenAI API配置模板
请复制此文件为 openai_config.py 并填入您的API密钥
"""

# OpenAI API配置
OPENAI_API_KEY = "your-openai-api-key-here"  # 替换为您的OpenAI API密钥
OPENAI_API_BASE = "https://api.openai.com/v1"  # 如果使用代理，请修改此URL

# 模型配置
VISION_MODEL = "gpt-4-vision-preview"  # 或 "gpt-4o" (支持视觉的模型)
MAX_TOKENS = 300  # 回答的最大token数

# 图像分析提示词
IMAGE_ANALYSIS_PROMPT = "请详细描述这张图片中看到的内容，包括物体、场景、颜色等信息。"

def get_openai_config():
    """获取OpenAI配置"""
    return {
        "api_key": OPENAI_API_KEY,
        "api_base": OPENAI_API_BASE,
        "model": VISION_MODEL,
        "max_tokens": MAX_TOKENS,
        "prompt": IMAGE_ANALYSIS_PROMPT
    } 