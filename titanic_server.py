import json
import joblib
import pandas as pd
import redis
import pika
import torch
import io
import hashlib
from minio import Minio

from model import NeuralNetwork

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # False -> without HTTPS
)

bucket_name = "titanic"

response = client.get_object(bucket_name, "titanic-model")
state_dict = torch.load(io.BytesIO(response.read()))
model = NeuralNetwork(input_size=8)
model.load_state_dict(state_dict)
model.eval()
response.close()
response.release_conn()

response = client.get_object(bucket_name, "titanic-scaler")
scaler = joblib.load(io.BytesIO(response.read()))
response.close()
response.release_conn()

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='queue.diia.titanic.req')
channel.queue_bind(exchange='titanic',
                   queue='queue.diia.titanic.req',
                   routing_key='queue.diia.titanic.req')


pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)


def get_info(info):

    feature_order = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]
    x = pd.DataFrame([info], columns=feature_order)

    x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    pred = model(x_tensor)
    prob = torch.sigmoid(pred).item()
    return prob

def get_data_with_cache(key, data_fetch_function, expiry_seconds=20):

    cached_data = r.get(key)
    if cached_data:
        print(f"'{key}' from cache.")
        return json.loads(cached_data)
    else:
        data = data_fetch_function()
        r.setex(key, expiry_seconds, json.dumps(data))
        return data

def on_request(ch, method, props, body):
    n = json.loads(body)

    data_json = json.dumps(n, sort_keys=True)
    key = hashlib.md5(data_json.encode()).hexdigest()

    print(f" [.] person info({n})")
    response = get_data_with_cache(key, lambda:get_info(n))

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=json.dumps(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='queue.diia.titanic.req', on_message_callback=on_request)


print(" [x] Awaiting RPC requests")
channel.start_consuming()