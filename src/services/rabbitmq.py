"""RabbitMQ publisher and consumer."""

import json
from typing import Awaitable, Callable, Optional

from aio_pika import DeliveryMode, Message, connect_robust

from src.config import get_settings

settings = get_settings()


class RabbitMQ:
    def __init__(self) -> None:
        self.connection = None
        self.channel = None

    async def connect(self) -> None:
        self.connection = await connect_robust(settings.rabbitmq_url)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=1)
        await self.channel.declare_queue(settings.rabbitmq_queue, durable=True)

    async def disconnect(self) -> None:
        if self.connection and not self.connection.is_closed:
            await self.connection.close()

    async def publish(self, job_id: str) -> None:
        if self.channel is None:
            await self.connect()
        await self.channel.default_exchange.publish(
            Message(
                json.dumps({"job_id": job_id}).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=settings.rabbitmq_queue,
        )

    async def consume(self, handler: Callable[[str], Awaitable[None]]) -> None:
        if self.channel is None:
            await self.connect()
        queue = await self.channel.declare_queue(settings.rabbitmq_queue, durable=True)
        async with queue.iterator() as messages:
            async for message in messages:
                async with message.process(requeue=False):
                    body = json.loads(message.body)
                    await handler(body["job_id"])


_rabbit: Optional[RabbitMQ] = None


async def get_rabbitmq() -> RabbitMQ:
    global _rabbit
    if _rabbit is None:
        _rabbit = RabbitMQ()
        await _rabbit.connect()
    return _rabbit
