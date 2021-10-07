import warnings
import sys
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler

warnings.filterwarnings('ignore')


class Predictor(object):
    """
    Simple class wrapping the inference procedure
    for a finetuned distilBERT model.
    """
    def __init__(self,
                 model_name,
                 tokenizer_name,
                 state_dict_path,
                 labels,
                 data_path,
                 out_path,
                 batch_size):
        super(Predictor, self).__init__()
        self.model_name: str = model_name
        self.tokenizer_name: str = tokenizer_name
        self.state_dict_path: str = state_dict_path
        self.data_path: str = data_path
        self.out_path: str = out_path
        self.batch_size: int = batch_size
        self.labels: dict = labels
        self.model = None
        self.tokenizer = None
        self.device = None

    def __inference(self, dataloader):
        self.model.eval()
        predictions = []
        ids = []
        for batch in dataloader:
            batch = tuple(item.to(self.device) for item in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            with torch.inference_mode():
                outputs = self.model(**inputs)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)
            ids.append(batch[2].detach().cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        ids = np.concatenate(ids, axis=0)
        predictions = np.argmax(predictions, axis=1).flatten()
        pd.DataFrame({'id': ids, 'preds': predictions}).to_csv(self.out_path,
                                                               mode='a',
                                                               index=False,
                                                               header=False)

    def __get_assets(self):
        self.tokenizer = (DistilBertTokenizerFast
                          .from_pretrained(self.tokenizer_name,
                                           do_lower_case=True))
        self.model = (DistilBertForSequenceClassification.
                      from_pretrained(self.model_name,
                                      num_labels=len(self.labels),
                                      output_attentions=False,
                                      output_hidden_states=False))
        self.model.load_state_dict(
            torch.load(self.state_dict_path,
                       map_location=torch.device('cpu')))
        self.device = (torch.device('cuda' if
                                    torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def __prepare_data(self, chunk):
        encoded_data = self.tokenizer(
            chunk.full_text.to_list(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=256,
            return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        ids = (torch.tensor(chunk.index.values))
        tf_dataset = TensorDataset(input_ids, attention_masks, ids)
        dataloader = DataLoader(tf_dataset,
                                sampler=SequentialSampler(tf_dataset),
                                batch_size=self.batch_size)
        return dataloader

    def run(self):
        print('Collecting assets... ', file=sys.stderr)
        self.__get_assets()
        print('[DONE]', file=sys.stderr)

        with pd.read_csv(self.data_path,
                         usecols=['full_text'],
                         lineterminator='\n',
                         chunksize=1000000, dtype=str) as reader:
            for chunk in reader:
                chunk.dropna(inplace=True)
                mydataloader = self.__prepare_data(chunk)
                self.__inference(mydataloader)
