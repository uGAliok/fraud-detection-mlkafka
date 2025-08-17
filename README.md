# Real-Time Fraud Detection System

Датасеты предоставлены в рамках соревнования [TETA ML 1. 2025](https://www.kaggle.com/competitions/teta-ml-1-2025)

Система для обнаружения мошеннических транзакций в реальном времени с использованием предобученной `LightGBM` модели и Kafka для потоковой обработки данных.

## Архитектура проекта
```
.
├── fraud_detector/
│   ├── src/
│   │   ├── preprocessing.py    # Пайплайн обработки данных
│   │   └── scorer.py           # Модуль прогнозирования
|   ├── models/
|   │   └── mylightgbm_model.pkl # Предобученная LightGBM модель
│   ├── app.py                  # Kafka Consumer/Producer
│   ├── Dockerfile
|   └── train_data/
|       └── train.csv # Данные для обучения (reference)
├── interface/
│   ├── app.py              # Streamlit UI
|   └── Dockerfile
├── docker-compose.yaml
├── .gitignore
└── README.md
```

## Компоненты системы:
### 1. interface (Streamlit UI):
   
   Создан для удобной симуляции потоковых данных с транзакциями. Реальный продукт использовал бы прямой поток данных из других систем.
    - Имитирует отправку транзакций в Kafka через CSV-файлы.
    - Генерирует уникальные ID для транзакций.
    - Загружает транзакции отдельными сообщениями формата JSON в топик kafka `transactions`.
    

### 2. fraud_detector (ML Service):
   - Загружает предобученную модель LightGBM (`mylightgbm_model.pkl`).
   - Выполняет препроцессинг данных и производит скоринг:
   - **1) Пайплайн обработки данных (`preprocessing.py`)**
      - Извлечение  года, месяца, дня, дня недели и часа
      - Парсинг `transaction_time` с обработкой некорректных значений
      - Вычисление расстояния между пользователем и мерчантом по формуле гаверсинуса
      - Используется `LabelEncoder`, обученный на train.csv
      - Поддержка `<UNK>` для неизвестных категорий
      - Логарифмирование `(log(x + 1))` для сглаживания распределения
     
   - **2) Модельный слой (`scorer.py`)**
      - Порог классификации: 0.686 (оптимально подобран)
      - Автоматическая загрузка модели при инициализации
      - Выгружает результат скоринга в топик kafka `scoring`

### 3. Kafka Infrastructure:
   - Zookeeper + Kafka брокер
   - `kafka-setup`: автоматически создает топики `transactions` и `scoring`
   - Kafka UI: веб-интерфейс для мониторинга сообщений (порт 8080)

### 4. Логирование
- `/app/logs/service.log` — основной лог-файл
- `stdout` — дублирование логов в консоль при запуске контейнера
- Уровни:
  - `INFO`: Основные этапы обработки
  - `DEBUG`: Детали преобразования данных
  - `ERROR`: Критические сбои с трейсбэками

## 🚀 Быстрый старт

### Требования
- Docker 20.10+
- Docker Compose 2.0+

### Запуск
```bash
git clone https://github.com/your-repo/fraud-detection-mlkafka.git
cd fraud-detection-mlkafka

# Сборка и запуск всех сервисов
docker-compose up --build
```
После запуска:
- **Streamlit UI**: http://localhost:8501
- **Kafka UI**: http://localhost:8080
- **Логи сервисов**: 
  ```bash
  docker-compose logs <service_name>  # Например: fraud_detector, kafka, interface

## 🛠️ Использование

### 1. Загрузка данных:

 - Загрузите CSV через интерфейс Streamlit. Для тестирования работы проекта используется файл формата `test.csv` из соревнования https://www.kaggle.com/competitions/teta-ml-1-2025
 - Пример структуры данных:
    ```csv
    transaction_time,amount,lat,lon,merchant_lat,merchant_lon,gender,...
    2023-01-01 12:30:00,150.50,40.7128,-74.0060,40.7580,-73.9855,M,...
    ```
 - Для первых тестов рекомендуется загружать небольшой семпл данных (до 100 транзакций) за раз, чтобы исполнение кода не заняло много времени.

### 2. Мониторинг:
 - **Kafka UI**: Просматривайте сообщения в топиках transactions и scoring
 - **Логи обработки**: /app/logs/service.log внутри контейнера fraud_detector

### 3. Результаты:

 - Скоринговые оценки пишутся в топик scoring в формате:
    ```json
    {
    "score": 0.995, 
    "fraud_flag": 1, 
    "transaction_id": "d6b0f7a0-8e1a-4a3c-9b2d-5c8f9d1e2f3a"
    }
    ```

## Настройки Kafka
```yml
Топики:
- transactions (входные данные)
- scoring (результаты скоринга)

Репликация: 1 (для разработки)
Партиции: 3
```

*Примечание:* 

Для полной функциональности убедитесь, что:
1. Модель `mylightgbm_model.pkl` размещена в `fraud_detector/models/`
2. Тренировочные данные находятся в `fraud_detector/train_data/`
3. Порты 8080, 8501 и 9095 свободны на хосте
4. Обработка происходит на CPU
