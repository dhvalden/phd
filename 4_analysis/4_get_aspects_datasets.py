import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-c', '--confidence', type=float)
    parser.add_argument('-m', '--min_events', type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, index_col=0)
    df = df.set_index(pd.to_datetime(df.date.values))
    df = df.drop(columns=['Unnamed: 0', 'date', 'is_retweet',
                          'full_text', 'user', 'clean_text'])

    fltr = df.iloc[:, :4] >= args.confidence  # Section with emotion values
    df2 = df[fltr.any(1)]
    df_b = df2.iloc[:, :4].idxmax(axis='columns')
    df_c = df2.iloc[:, 4:]

    df_final = pd.concat([df_c, pd.get_dummies(df_b)],
                         axis=1)

    df_gbm = df_final.loc[(df_final['groupbased-motivator'] > args.min_events)]
    df_ca = df_final.loc[(df_final['collective-action'] > args.min_events)]
    df_in = df_final.loc[(df_final['ingroup'] > args.min_events)]
    df_out = df_final.loc[(df_final['outgroup'] > args.min_events)]

    prefix = os.path.splitext('/path/to/some/file.txt')[0]

    df_gbm.to_csv(f'{prefix}_gbm.csv')
    df_ca.to_csv(f'{prefix}_gbm.csv')
    df_in.to_csv(f'{prefix}_gbm.csv')
    df_out.to_csv(f'{prefix}_gbm.csv')


if __name__ == '__main__':
    main()
