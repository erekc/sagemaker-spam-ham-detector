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
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
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
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
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
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
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
