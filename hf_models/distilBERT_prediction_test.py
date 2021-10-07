import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler


def predict(dataloader_pred, model, device):
    model.eval()
    predictions = []
    for batch in dataloader_pred:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def runner(batch_size, data_path, model_path):
    # Initial Set up
    model = (DistilBertForSequenceClassification
             .from_pretrained("distilbert-base-uncased",
                              num_labels=2,
                              output_attentions=False,
                              output_hidden_states=False))
    tokenizer = (DistilBertTokenizerFast
                 .from_pretrained('distilbert-base-uncased',
                                  do_lower_case=True))
    df = pd.read_csv(data_path, names=['text'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device('cpu')))
    encoded_data_pred = tokenizer(
        df.text.to_list(),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt')

    input_ids_pred = encoded_data_pred['input_ids']
    attention_masks_pred = encoded_data_pred['attention_mask']
    dataset_pred = TensorDataset(input_ids_pred, attention_masks_pred)
    dataloader_pred = DataLoader(dataset_pred,
                                 sampler=SequentialSampler(dataset_pred),
                                 batch_size=batch_size)
    results = pd.Series(predict(dataloader_pred, model, device))
    results.to_csv('results.csv')


if __name__ == '__main__':
    runner(8, 'data/fr_data.csv',
           'trained_models/fr_distilBERT_epoch_4.model')
