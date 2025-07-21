# import numpy as np
import pandas as pd

df1 = pd.read_csv('/Users/angwang-yun/Desktop/Project/comment_emotion/keyword연금보험.csv', encoding='utf-8-sig')
breakpoint()
df2 = pd.read_csv('/Users/angwang-yun/Desktop/Project/comment_emotion/keyword상해보험.csv', encoding='utf-8-sig')
breakpoint()
print(df1.shape, "\n",df2.shape)
merged_df = pd.concat([df1, df2], ignore_index=True)
breakpoint()

merged_df.to_csv('merged_keywords_연금_상해.csv', index=False)