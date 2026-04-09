"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any, Dict

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在 Pydantic Settings 初始化之前，必须先加载 .env 文件
# 否则 pydantic_settings 会读取不到环境变量（虽然 .env 存在）
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OnCallAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    # 国内百炼 Key 须用北京域名；langchain_qwq.ChatQwen 默认是国际站，必须通过配置显式指定
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    @field_validator("dashscope_api_key", "dashscope_api_base", mode="before")
    @classmethod
    def strip_secrets(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 意图识别配置
    intent_enabled: bool = True                   # 是否启用意图识别（关闭时走原有关键词兜底）
    intent_llm_enabled: bool = False               # 是否用 LLM 做意图分类（False 则用关键词兜底）
    intent_min_score: float = 0.3                  # 意图分数下限，低于此分过滤掉
    max_intent_per_subquestion: int = 5            # 单个子问题的最大意图数量
    max_total_intents: int = 10                   # 所有子问题的意图总上限
    operation_confirm_threshold: float = 0.7       # operation 类意图需达到此分数才触发二次确认（防误触发）

    # 多问句拆分配置
    multi_question_split_enabled: bool = False     # 是否启用多问句拆分（关闭则走单问句逻辑）
    query_rewrite_enabled: bool = False           # 是否用 LLM 做问句改写（关闭则用规则拆分）

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 阿里云 OCR 配置
    aliyun_access_key_id: str = ""       # 阿里云 AccessKey ID
    aliyun_access_key_secret: str = ""    # 阿里云 AccessKey Secret
    aliyun_region_id: str = "cn-shanghai" # 阿里云 Region

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
