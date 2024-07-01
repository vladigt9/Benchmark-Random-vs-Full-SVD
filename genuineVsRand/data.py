import pandas as pd
import numpy as np


def createHiggsData():
    df = pd.read_csv('data/HIGGS/HIGGS.csv', names=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                                     21,22,23,24,25,26,27,28,29])

    df[30] = df[29]*df[28]
    df[31] = df[29]*df[27]
    df[32] = df[29]*df[26]
    df[33] = df[29]*df[25]
    df[34] = df[29]*df[24]
    df[35] = df[29]*df[23]
    df[36] = df[29]*df[22]
    df[37] = df[29]*df[21]
    df[38] = df[29]*df[20]
    df[39] = df[29]*df[11]
    df[40] = df[29]*df[12]
    df[41] = df[29]*df[13]
    df[42] = df[29]*df[14]
    df[43] = df[29]*df[15]
    df[44] = df[29]*df[16]
    df[45] = df[29]*df[17]
    df[46] = df[29]*df[18]
    df[47] = df[29]*df[19]
    df[48] = df[29]*df[13]
    df[49] = df[29]**df[14]
    df[50] = df[29]**df[15]
    df[51] = df[29]**df[16]
    df[52] = df[29]**df[17]
    df[53] = df[29]**df[18]
    df[54] = df[29]**df[19]
    df = df.dropna()
    df = df.drop(1, axis=1)

    return df


def createHconsData():
    df = pd.read_csv('data/hcons/hcons.csv')
    
    df = df.dropna()

    mean = 0
    std_dev = 0.1
    noise1 = np.random.normal(mean, std_dev, df['1'].shape)
    noise2 = np.random.normal(mean, std_dev, df['2'].shape)
    noise3 = np.random.normal(mean, std_dev, df['3'].shape)
    noise4 = np.random.normal(mean, std_dev, df['4'].shape)

    df['5'] = df['1']*df['2']
    df['6'] = df['1']*df['3']
    df['7'] = df['1']*df['4']
    df['8'] = df['1']*np.cos(df['2'])
    df['9'] = df['1']*np.cos(df['3'])
    df['10'] = df['1']*np.cos(df['4'])
    df['11'] = df['1']/df['2']
    df['12'] = df['1']/df['3']
    df['13'] = df['1']/df['4']
    df['14'] = df['1']*noise1
    df['15'] = df['1']*noise2
    df['16'] = df['1']*noise3
    df['17'] = df['1']*noise4
    df['18'] = df['1']*np.sqrt(df['2'])
    df['19'] = df['1']*np.sqrt(df['3'])
    df['20'] = df['1']*np.sqrt(df['4'])
    df['21'] = df['2']*df['2']
    df['22'] = df['2']*df['3']
    df['23'] = df['2']*df['4']
    df['24'] = df['2']*np.cos(df['2'])
    df['25'] = df['2']*np.cos(df['3'])
    df['26'] = df['2']*np.cos(df['4'])
    df['27'] = df['2']/df['2']
    df['28'] = df['2']/df['3']
    df['29'] = df['2']/df['4']
    df['30'] = df['2']*noise1
    df['31'] = df['2']*noise2
    df['32'] = df['2']*noise3
    df['33'] = df['2']*noise4
    df['34'] = df['2']*np.sqrt(df['2'])
    df['35'] = df['2']*np.sqrt(df['3'])
    df['36'] = df['2']*np.sqrt(df['4'])
    df['37'] = df['3']*df['2']
    df['38'] = df['3']*df['3']
    df['39'] = df['3']*df['4']
    df['40'] = df['3']*np.cos(df['2'])
    df['41'] = df['3']*np.cos(df['3'])
    df['42'] = df['3']*np.cos(df['4'])
    df['43'] = df['3']/df['2']
    df['44'] = df['3']/df['3']
    df['45'] = df['3']/df['4']
    df['46'] = df['3']*noise1
    df['47'] = df['3']*noise2
    df['48'] = df['3']*noise3
    df['49'] = df['3']*noise4
    df['50'] = df['3']*np.sqrt(df['2'])
    df['51'] = df['3']*np.sqrt(df['3'])
    df['52'] = df['3']*np.sqrt(df['4'])
    df=df.dropna()

    return df
