import os
import sys
import pandas as pd
import time
import json
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
from confluent_kafka import Consumer, Producer, KafkaError
# чтобы видеть модули из корня
sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, model as lgbm_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")

class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest'
        }
        self.producer_config = {
             'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS
             }
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(self.producer_config)
        
        self.train, self.encoders = load_train_data()
        logger.info('Service initialized')
    
    def _send(self, topic: str, payload: dict):
        try:
            self.producer.produce(topic, value=json.dumps(payload).encode('utf-8'))
            self.producer.poll(0)
        except BufferError:
            # если буфер переполнен
            self.producer.flush(2.0)
            self.producer.produce(topic, value=json.dumps(payload).encode('utf-8'))

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            try:
                # Десериализация JSON
                data = json.loads(msg.value().decode('utf-8'))

                # Извлекаем ID и данные
                transaction_id = data['transaction_id']
                input_df = pd.DataFrame([data['data']])

                # Препроцессинг и предсказание
                processed_df = run_preproc(self.train, input_df, self.encoders)
                scores, preds = make_pred(lgbm_model,processed_df)

                # Отправка результата в топик scoring
                payload = {
                    "transaction_id": transaction_id,
                    "prediction": int(preds[0]),
                    "probability": float(scores[0])
                }
                self._send(SCORING_TOPIC, payload)

                self.producer.flush()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                time.sleep(0.1)
        self.producer.flush()

if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        service.producer.flush()