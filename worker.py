import uuid

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.middleware import Middleware
from text_to_image import TexttoImage

class TextToImageMiddleware(Middleware):
    def __init__(self) -> None:
        super().__init__()
        self.text_to_image = TexttoImage()
    
    def after_process_boot(self, broker):
        self.text_to_image.load_model()
        return super().after_process_boot(broker)

text_to_image_middleware = TextToImageMiddleware()
redis_broker = RedisBroker(host="localhost")
redis_broker.add_middleware(text_to_image_middleware)
dramatiq.set_broker(redis_broker) 
