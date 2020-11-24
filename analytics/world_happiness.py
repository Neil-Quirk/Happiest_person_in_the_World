import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_2015 = pd.read_csv(r'C:\Users\neilr\Documents\Workspace\Happiest_person_in_the_World\datasets\world_happiness_index\2015.csv')

df_2016 = pd.read_csv(r'C:\Users\neilr\Documents\Workspace\Happiest_person_in_the_World\datasets\world_happiness_index\2016.csv')

df_2017 = pd.read_csv(r'C:\Users\neilr\Documents\Workspace\Happiest_person_in_the_World\datasets\world_happiness_index\2017.csv')

df_2018 = pd.read_csv(r'C:\Users\neilr\Documents\Workspace\Happiest_person_in_the_World\datasets\world_happiness_index\2018.csv')

df_2019 = pd.read_csv(r'C:\Users\neilr\Documents\Workspace\Happiest_person_in_the_World\datasets\world_happiness_index\2019.csv')

df1 = df_2015.copy()
df2 = df_2016.copy()
df3 = df_2017.copy()
df4 = df_2018.copy()
df5 = df_2019.copy()

df1.drop(['Standard Error', 'Freedom', 'Trust (Government Corruption)',
       'Generosity', 'Dystopia Residual', 'Region', 'Family'], axis=1, inplace = True)

df2.drop(['Lower Confidence Interval', 'Upper Confidence Interval',
       'Freedom', 'Trust (Government Corruption)', 'Generosity',
       'Dystopia Residual', 'Region', 'Family'], axis = 1, inplace =True)

df3.drop(['Whisker.high',
       'Whisker.low', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual', 'Family'], axis = 1, inplace= True)

df4.drop(['Social support', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption'], axis = 1 , inplace = True)

df5.drop(['Social support','Freedom to make life choices', 'Generosity',
       'Perceptions of corruption'], axis = 1, inplace = True)

df1 = df1.assign(Year = 2015)
df2 = df2.assign(Year = 2016)
df3 = df3.assign(Year = 2017)
df4 = df4.assign(Year = 2018)
df5 = df5.assign(Year = 2019)

column_names = ['Country or region', 'Overall rank', 'Score', 'GDP per capita','Healthy life expectancy', 'Year']
df4 = df4.reindex(columns = column_names)
column_names = ['Country or region', 'Overall rank', 'Score', 'GDP per capita','Healthy life expectancy', 'Year']
df5 = df5.reindex(columns = column_names)

corrected_columns = ['Location', 'Rank', 'Happiness Score', 'GDP per Capita', 'Life Expectancy', 'Year']
df1.columns = corrected_columns
df2.columns = corrected_columns
df3.columns = corrected_columns
df4.columns = corrected_columns
df5.columns = corrected_columns

df0 = pd.concat([df1, df2, df3, df4, df5], axis=0)
df0.reset_index(level=0, drop=True, inplace=True)
df0



lm = LinearRegression()

# Linear Regression line
Z = df0[['GDP per Capita']]
lm.fit(Z, df0['Happiness Score'])
lm.intercept_
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="GDP per Capita", y="Happiness Score", data=df0)
plt.ylim(0,)


# Correlation Heatmat
corr = df0.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

pivot_df = pd.pivot_table(df0, index =['Location'], values=['Happiness Score'], aggfunc=np.mean)
pivot_df.sort_values(by='Happiness Score', ascending=False, inplace=True)

print(pivot_df)