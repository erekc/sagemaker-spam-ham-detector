import json
import boto3
import email
from email.parser import BytesParser, Parser
import os
import string
from hashlib import md5
import sys
sys.path.insert(1, '/opt')
import numpy as np

"""
The following encoding functions were taken from the AWS SageMaker spam-ham tutorial.
"""
if sys.version_info < (3,):
    maketrans = str.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):

    if lower:
        text = text.lower()

    translate_dict = dict((c, split) for c in filters)
    translate_map = maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):

    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
                      
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

def lambda_handler(event, context):
    email_key = event["Records"][0]["s3"]["object"]["key"]
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    
    s3 = boto3.client("s3")
    get_object_response = s3.get_object(
        Bucket=bucket,
        Key=email_key
    )
    
    email_body = get_object_response["Body"].read()
    
    message = email.message_from_bytes(email_body)
    
    return_email = message["Return-Path"][1:-1]
    print(return_email)
    receive_date = message["Date"]
    print(receive_date)
    received_subject = message["Subject"]
    print(received_subject)
    
    message_text = ""
    for part in message.walk():
        if part.get_content_type() == "text/plain":
            message_part = part.as_string().split("\n")
            message_list = message_part[2:-1]
            message_text = " ".join(message_list)
    print(message_text)
    sample_message = []
    count = 0
    while count < 240 and count < len(message_text):
        sample_message.append(message_text[count])
        count += 1
    sample_message = ''.join(sample_message)
    print(sample_message)
    
    input_messages = [message_text]
    one_hot_test_messages = one_hot_encode(input_messages, 9013)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, 9013)
    
    model_name = os.environ["ModelName"]
    print(model_name)
    sagemaker = boto3.client("sagemaker-runtime")
    sagemaker_response = sagemaker.invoke_endpoint(
        EndpointName=model_name,
        Body=json.dumps(encoded_test_messages.tolist()),
        ContentType="application/json",
    )
    print(sagemaker_response)
    sagemaker_body = sagemaker_response["Body"]
    print(sagemaker_body)
    sagemaker_message = sagemaker_body.read()
    print(sagemaker_message)
    
    response_json = json.loads(sagemaker_message)
    print(response_json)
    
    predicted_label = response_json["predicted_label"][0][0]
    predicted_probability = response_json["predicted_probability"][0][0]
    label = "ham" if predicted_label == 0.0 else "spam"
    confidence = predicted_probability*100
    
    print("Predicted")
    
    ses = boto3.client("ses")
    send_destination = {
        "ToAddresses": [
            return_email
        ]
    }
    
    message_to_send = """
    We received your email sent at {} with the subject {}.\n\n
    Here is a 240 character sample of the email body:\n\n
    {}\n\n
    The email was categorized as {} with a {}% confidence.
    """.format(receive_date, received_subject, sample_message, label, confidence)
    print(message_to_send)
    
    send_message = {
        "Subject": {
            "Data": "Spam Protection Services"
        },
        "Body": {
            "Text": {
                "Data": message_to_send
            },
        }
    }
    
    ses.send_email(
        Source="spamham@erekryo.software",
        Destination=send_destination,
        Message=send_message
    )
    print("Email sent to {}".format(return_email))
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
