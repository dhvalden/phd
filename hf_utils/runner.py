from hf_utils.predictor import Predictor


def main():
    mypredictor = Predictor(
        model_name='distilbert-base-uncased',
        tokenizer_name='distilbert-base-uncased',
        state_dict_path='./trained_models/emo_distilBERT_epoch_4.model',
        labels={'sadness': 0, 'anger': 1, 'fear': 2, 'joy': 3},
        data_path='../rawdata/all_climate.csv',
        out_path='../rawdata/labels_all_climate.csv',
        batch_size=32
    )
    mypredictor.run()


if __name__ == '__main__':
    main()
