import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--df_text', type=str)
    parser.add_argument('-p', '--df_probs', type=str)
    parser.add_argument('-o', '--output_file', type=str)
    args = parser.parse_args()
    df1 = pd.read_csv(args.df_text,
                      lineterminator='\n',
                      usecols=['date',
                               'full_text',
                               'is_retweet',
                               'user'])
    df2 = pd.read_csv(args.df_probs,
                      lineterminator='\n',
                      index_col='id')

    labels = {'sadness': '0', 'anger': '1', 'fear': '2', 'joy': '3'}
    inverted_labels = {v: k for k, v in labels.items()}
    df2.rename(columns=inverted_labels, inplace=True)

    print(df1)
    print(df2)

    df3 = pd.concat([df1, df2], axis=1)

    print(df3)

    df3.to_csv(args.output_file)


if __name__ == '__main__':
    main()
