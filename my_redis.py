import json
import time
from typing import Generator
import redis
import requests

# 配置信息
REDIS_HOST = "localhost"
REDIS_PORT = 6379
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:8b"


# 初始化Redis连接
class RedisManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加消息到聊天历史"""
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        # 使用管道操作保证原子性
        with self.redis_client.pipeline() as pipe:
            pipe.rpush(f"chat_session:{session_id}", message)
            # 保留最近20条消息防止内存溢出
            pipe.ltrim(f"chat_session:{session_id}", -20, -1)
            pipe.execute()

    def get_history(self, session_id: str, max_messages: int = 5) -> list:
        """获取最近的聊天历史"""
        messages = self.redis_client.lrange(
            f"chat_session:{session_id}",
            -max_messages,
            -1
        )
        return [json.loads(msg) for msg in messages]


class RAGService:
    def __init__(self):
        self.redis_manager = RedisManager()

    def _format_prompt(self, history: list, query: str) -> str:
        """将历史记录格式化为模型输入的prompt"""
        context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history]
        )
        return f"{context}\nUser: {query}\nAI:"

    def stream_generate(
            self,
            session_id: str,
            query: str,
            max_history: int = 5
    ) -> Generator[str, None, None]:
        # 保存用户提问
        self.redis_manager.add_message(session_id, "user", query)

        # 获取历史上下文
        history = self.redis_manager.get_history(session_id, max_history)
        prompt = self._format_prompt(history, query)

        # 准备流式请求
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "context": self._get_last_context(session_id)  # 可选的上下文维持
            },
            stream=True
        )

        full_response = []
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    if not chunk.get("done"):
                        token = chunk.get("response", "")
                        full_response.append(token)
                        yield token  # 流式输出token

            # 完整响应保存到历史记录
            self.redis_manager.add_message(
                session_id,
                "assistant",
                "".join(full_response)
            )

        except Exception as e:
            yield f"[生成错误] {str(e)}"
            raise

    def _get_last_context(self, session_id: str) -> list:
        """获取上次对话的上下文（适用于需要维持对话状态的模型）"""
        # 这里根据具体模型的上下文需求实现
        # 示例返回空列表，实际可能需要存储模型返回的context
        return []


# 使用示例
if __name__ == "__main__":
    service = RAGService()
    session_id = "test_session_123"  # 实际应用中应生成唯一ID


    # 模拟对话流程
    def simulate_chat():
        while True:
            query = input("User: ")
            if query.lower() == 'exit':
                break

            print("AI: ", end="", flush=True)
            for chunk in service.stream_generate(session_id, query):
                print(chunk, end="", flush=True)
            print("\n")


    simulate_chat()