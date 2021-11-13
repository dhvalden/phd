from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

top_N = 60

df = pd.read_csv('~/code/phd/data/climate_pos_f4f.csv')
df


words = (df.nouns
           .str.lower()
           .str.cat(sep=' ')
           .split()
)

words

# generate DF out of Counter
rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')
print(rslt)

# plot
rslt.plot.bar(rot=90, figsize=(16,10), width=0.8)
plt.show()
