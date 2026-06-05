import pandas as pd

df = pd.read_csv('output/submission.csv')
df = df.sort_values(['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
df['rank'] = range(1, 101)
df.to_csv('output/submission.csv', index=False)
print('Fixed!')