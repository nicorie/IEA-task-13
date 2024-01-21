# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:42:27 2023

survey analysis

@author: nri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

PATH = r'C:\Users\NRI\OneDrive - EE\Skrivebord\IEA PVPS\Task 13\Survey'
FILE = '\Bifi-Tracker-Survey-Summaries.xlsx'

df = pd.read_excel(PATH+FILE, skiprows=1)

df.drop([0], inplace=True)
del_cols = [i for i in df.columns if 'Unnamed' in str(i)]
df.drop(columns=del_cols, inplace=True)

df = df.T
questions = list(df.iloc[0].values)
df.columns = questions
df.drop(['Question'], inplace=True)

# %% certification

Qs_cert = ['Do your trackers have any certifications?',
           'How many years selling trackers',
           'How many countries do you have sales in?',
           'Number of projects (in GW) finished',
           'Number of patents',
           'How are the trackers moved? How many rows per motor?']

df_cert = df[Qs_cert]
df_cert = df_cert[df_cert[Qs_cert[0]].notna()]

df_cert['IEC'] = df[Qs_cert[0]].str.contains('IEC')
df_cert['UL'] = df[Qs_cert[0]].str.contains('UL')
df_cert['3rd_Party'] = df[Qs_cert[0]].str.contains('B&V|VDE|DNV')

hue_order = ['IEC and UL and 3rd Party', 'IEC and UL', 'IEC Only', 'UL Only',
             'IEC and 3rd Party', 'UL and 3rd Party', '3rd Party Only', 'None']


def certGroups(df):
    if (df['IEC']) and (df['UL']) and (df['3rd_Party']):
        return 'IEC and UL and 3rd Party'
    if (df['IEC']) and (df['UL']) and ~(df['3rd_Party']):
        return 'IEC and UL'
    if (df['IEC']) and ~(df['UL']) and (df['3rd_Party']):
        return 'IEC and 3rd Party'
    if (df['IEC']) and ~(df['UL']) and ~(df['3rd_Party']):
        return 'IEC Only'
    if ~(df['IEC']) and (df['UL']) and (df['3rd_Party']):
        return 'UL and 3rd Party'
    if ~(df['IEC']) and (df['UL']) and ~(df['3rd_Party']):
        return 'UL Only'
    if ~(df['IEC']) and ~(df['UL']) and (df['3rd_Party']):
        return '3rd Party Only'
    else:
        return 'None'


df_cert.rename(columns={Qs_cert[2]: 'Number of countries with sales',
                        Qs_cert[1]: 'Years selling trackers',
                        Qs_cert[3]: 'Number of projects finished (GW)'},
               inplace=True)

df_cert['Certifications'] = df_cert.apply(certGroups, axis=1)

min_s = min(df_cert['Years selling trackers'])
max_s = max(df_cert['Years selling trackers'])
p = sns.relplot(data=df_cert, x='Number of projects finished (GW)',
                y='Number of countries with sales', hue='Certifications',
                hue_order=hue_order, size='Years selling trackers',
                edgecolor=None, sizes=(min_s*5, max_s*8))
p.set(xscale='log')
p.set_ylabels(fontsize=14)
p.set_xlabels('Capacity of finished projects (GW)', fontsize=14)
plt.grid()

# %% tracker power

Qs_pwr = ['How are the trackers moved? ',
          'How many rows per motor?',
          'What is the maximum length of each row (m)?',
          'What is the maximum tilt angle?',
          'How are the tracker motors powered?',
          'Max N-S slope (%)', 'Max E-W slope (%)']

df_pwr = df[Qs_pwr]
df_pwr[Qs_pwr[1:4]] = df_pwr[Qs_pwr[1:4]].astype(float)
df_pwr[Qs_pwr[5:7]] = df_pwr[Qs_pwr[5:7]].astype(float)
# df_pwr = df_pwr[df_pwr[Qs_pwr[]].notna()]

# df_pwr[df_pwr[Qs_pwr[2]].notna()].boxplot(Qs_pwr[2])
# df_pwr[df_pwr[Qs_pwr[1]].notna()].boxplot(Qs_pwr[1])

df_pwr['Grid'] = df[Qs_pwr[4]].str.contains('Grid')
# df_pwr['PV String'] = df[Qs_pwr[3]].str.contains('PV String')
# df_pwr['PV Panel'] = df[Qs_pwr[3]].str.contains('PV Panel')
# here we don't know if it's a panel or string
df_pwr['PV'] = df[Qs_pwr[4]].str.contains('PV')

df_pwr.rename(columns={Qs_pwr[3]: 'Max tilt angle (°)'}, inplace=True)


def pwrGroups(df):
    if (df['Grid']) and (df['PV']):
        return 'Grid or PV'
    if (df['Grid']) and ~(df['PV']):
        return 'Grid Only'
    if ~(df['Grid']) and (df['PV']):
        return 'PV Only'


df_pwr['Tracker Power'] = df_pwr.apply(pwrGroups, axis=1)
df_pwr['Max Rows per Motor'] = df_pwr[Qs_pwr[1]].astype(str)
df_pwr['Max Rows per Motor'].dropna()

max_s = max(df_pwr['Max tilt angle (°)'])
p = sns.relplot(data=df_pwr, x=Qs_pwr[5], y=Qs_pwr[2],
                hue='Max Rows per Motor', style='Tracker Power',
                hue_order=['1.0', '2.0', '32.0'],
                s=150)

p.set_ylabels('Max Row Length (m)', fontsize=14)
p.set_xlabels(fontsize=14)
plt.grid()

# %% algorithms

Qs_alg = ['What options do customers have for different tracking algorithms?',
          'Number of projects (in GW) finished']

df_alg = df[Qs_alg]
df_alg = df_alg[df_alg[Qs_alg[0]].notna()]


def companySize(df):
    if (df[Qs_alg[1]]) >= 10:
        return '>10GW'
    elif (df[Qs_alg[1]]) >= 1:
        return '1–10GW'
    else:
        return '<1GW'


df_alg['Size'] = df_alg.apply(companySize, axis=1)

company_sizes = df_alg['Size'].value_counts()

alg_cnt, company, size = [[] for i in range(3)]

for i in range(len(df_alg)):
    algs = df_alg.iloc[i][0].split(',')
    co = df_alg.index[i]
    s = df_alg['Size'].iloc[i]
    for a in algs:
        alg_cnt.append(a.replace(' ', ''))
        company.append(co)
        size.append(s)

df_alg_stacked = pd.DataFrame(data=zip(company, size, alg_cnt), columns=[
                              'company', 'size', 'algorithms'])

data = df_alg_stacked['algorithms'].value_counts()/len(df_alg)
counts, algorithms = data.values, data.index

data.rename('counts', inplace=True)
data = pd.DataFrame(data)
data.reset_index(inplace=True)

# plt.figure()
# plt.grid(linestyle='dashed')
# plt.bar(algorithms, counts*100)
# plt.ylabel('Companies with algorithm (%)', fontsize=14)
# plt.show()

data = df_alg_stacked['algorithms'].groupby(
    df_alg_stacked['size']).value_counts()

data.rename('counts', inplace=True)
data = pd.DataFrame(data)
data.reset_index(inplace=True)

fig, ax = plt.subplots()

x = np.arange(len(data['algorithms'].unique()))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

for s in ['<1GW', '1–10GW', '>10GW']:
    sub = data.loc[data['size'] == s]
    size = company_sizes[s]
    offset = width * multiplier
    rects = ax.bar(x+offset, (sub['counts']/size)*100, width, label=s)
    multiplier += 1

ax.set_ylabel('Companies with algorithm (%)', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.set_xticks(x + width, data['algorithms'].unique(), rotation=0)
ax.legend(loc='upper right', title='Company Size', fontsize=12)
ax.grid(linestyle='dashed')

plt.show()

# %% bifacial

Qs_bifi = ['How many MW of trackers did you ship in 2020, 2021, and 2022?',
           'What fraction of those systems used bifacial modules?']

df_bifi = df[Qs_bifi]
df_bifi = df_bifi[(df_bifi[Qs_bifi[0]].notna()) &
                  (df_bifi[Qs_bifi[1]].notna())]


def splitYears(df_bifi_Q):
    data20, data21, data22, co = [], [], [], []
    for ct, i in enumerate(df_bifi_Q):
        data = i.split(',')
        data20.append(float(data[0]))
        data21.append(float(data[1]))
        data22.append(float(data[2]))
        co.append(str(df_bifi_Q.index[ct]))

    return data20, data21, data22, co


MW20, MW21, MW22, co = splitYears(df_bifi[Qs_bifi[0]])
BF20, BF21, BF22, co = splitYears(df_bifi[Qs_bifi[1]])

years = [2020]*len(MW20) + [2021]*len(MW21) + [2022]*len(MW22)
companies = co + co + co

BF = BF20 + BF21 + BF22
MW = MW20 + MW21 + MW22

df_bifi_plt = pd.DataFrame(data=zip(years, companies, MW, BF),
                           columns=['Year', 'Company', 'MW Deployed',
                                    '% Bifacial'])

df_bifi_sum = df_bifi_plt['MW Deployed'].describe()

p = sns.relplot(data=df_bifi_plt, x='Year', y='% Bifacial', hue='Company',
                size='MW Deployed', edgecolor=None, linewidth=0,
                sizes=(df_bifi_sum['25%']*0.05, df_bifi_sum['75%']*0.2))
# or you can define a size for every value in df_bifi_plt['MW Deployed']

p = sns.lineplot(data=df_bifi_plt, x='Year', y='% Bifacial',
                 hue='Company', legend=False)

#p.legend(fontsize=12, loc='best')
p.set_xlabel('Year', fontsize=14)
p.set_ylabel('% Bifacial (approximate)', fontsize=14)
p.tick_params(axis='both', labelsize=12)
p.set_xticks([2020, 2021, 2022])
p.grid(linestyle='dashed')

fig, axs = plt.subplots(nrows=3, sharex=True)

for i, year in enumerate(df_bifi_plt['Year'].unique()):
    sub = df_bifi_plt.loc[df_bifi_plt['Year'] == year]
    axs[i].hist(sub['% Bifacial'])
