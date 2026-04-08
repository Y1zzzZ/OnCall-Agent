from my_agent.core.milvus_manager import MilvusManager
from loguru import logger
import time

def test_connection():
    logger.info("开始测试 Milvus 连接 (v2 API)...")
    try:
        # 尝试初始化管理器，它会自动执行 _connect
        manager = MilvusManager(host="127.0.0.1", port="19530")
        
        # 检查 client 是否成功生成
        if manager.client:
            logger.success("✅ 服务器连接成功！")
            
            # 试着列出所有现有的集合 (Collection)
            collections = manager.client.list_collections()
            logger.info(f"当前库中存在的集合: {collections}")
            
            # 手动触发一下集合初始化测试
            manager.init_collection()
            logger.success("✅ 集合初始化/加载测试通过！")
        else:
            logger.error("❌ 客户端对象为空，请检查 pymilvus 安装。")
            
    except Exception as e:
        logger.error(f"❌ 连接测试失败: {e}")
        logger.warning("提示：请确保 docker compose up -d 已成功运行，且 19530 端口已映射。")

if __name__ == "__main__":
    test_connection()
