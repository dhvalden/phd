from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import ujson as json

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tweets = []

with open('./data/testing_sample.json', 'r') as f:
    for line in f:
        tweet = json.loads(line)
        tweets.append(tweet['full_text'])

inputs_list = []
tokens_list = []
outputs_list = []
predictions_list = []

for tweet in tweets:
    inputs = tokenizer(tweet, return_tensors="pt")
    tokens = inputs.tokens()
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    inputs_list.append(inputs)
    tokens_list.append(tokens)
    outputs_list.append(outputs)
    predictions_list.append(predictions)


#for token, prediction in zip(tokens, predictions[0].numpy()):
#    print((token, model.config.id2label[prediction]))
