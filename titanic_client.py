import pika
import uuid
import json
import sys


class TitanicClient(object):

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))

        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange='titanic', exchange_type='direct', durable=True)

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = json.loads(body.decode())

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='titanic',
            routing_key='queue.diia.titanic.req',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(n)
        )
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        return self.response


titanic = TitanicClient()

message = ' '.join(sys.argv[1:])
print(f" [x] Requesting info")
user_data = {"Pclass": 1, "Age": 15, "SibSp": 0, "Parch": 0, "Fare": 25.0, "Sex_male": 0, "Embarked_Q": 0, "Embarked_S": 0}
response = titanic.call(user_data)

print(f" [.] Got {response}")