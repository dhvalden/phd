import tensorflow as tf
import ujson as json
import argparse
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification


def labeler(args):
    input_path = args.input_path
    model = TFBertForSequenceClassification.from_pretrained("sent_imdb_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    labels = ['Negative', 'Positive']
    with open(input_path, 'r') as f:
        for line in f:
            output = {}
            tweet = json.loads(line)
            text = tweet['full_text']
            tf_batch = tokenizer(text,
                                 max_length=128,
                                 padding=True,
                                 truncation=True,
                                 return_tensors='tf')
            tf_outputs = model(tf_batch)
            tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
            label = tf.argmax(tf_predictions, axis=1)
            label = label.numpy()
            output['sentiments'] = labels[label[0]]
            output['score'] = str(label[0])
            output['date'] = tweet['date']
            output['full_text'] = tweet['full_text']
            print(json.dumps(output, ensure_ascii=False))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str,
                        help="Input file")
    args = parser.parse_args()

    labeler(args)

if __name__ == '__main__':
    main()
