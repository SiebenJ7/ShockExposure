#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:07:47 2024

@author: JuliusSiebenaller
"""
import numpy as np
import pandas as pd

dtafile = '/Users/JuliusSiebenaller/Documents/HSG/BVWL/BA/BA_JS/ZMKR51FL.DTA'

df = pd.read_stata(dtafile)
df.tail()
df.head()

# Missing values replacements (similar to Stata 'replace')
# Replace values using a dictionary
replace_dict = {
    'hw70': {9996: np.nan, 9997: np.nan, 9998: np.nan, 9999: np.nan},
    'hw71': {9996: np.nan, 9997: np.nan, 9998: np.nan, 9999: np.nan},
    'hw72': {9996: np.nan, 9997: np.nan, 9998: np.nan, 9999: np.nan},
    'hw2': {999: np.nan},
    'hw3': {9999: np.nan},
    'v437': {9999: np.nan},
    'v438': {9999: np.nan},
    'v131': {99: np.nan},
    'v463z': {9: np.nan},
    'm4': {96: np.nan, 97: np.nan, 98: np.nan, 99: np.nan, 94: 0}, # still breastfeeding, 0: never breastfed
    'h10': {8: np.nan, 9: np.nan},
    'v104': {97: np.nan, 98: np.nan, 99: np.nan}
}

df.replace(replace_dict, inplace=True)

# Handle breastfeeding and respondent residence separately
df.loc[df['m4'] == 95, 'm4'] = df['hw1']  # breastfeeding: assign age in months of child
df.loc[df['v104'] == 95, 'v104'] = df['v012']  # respondent has always lived in this place


# Create new variables using a dictionary for assignment
df = df.assign(
    resage=df['v012'],
    educlv=df['v106'],
    educyr=df['v107'],
    agehh=df['v152'],
    totchrbn=df['v201'],
    childnum=df['b16'],
    breastfd=df['m4'],
    reswgt=df['v437'] / 10,
    resht=df['v438'] / 10,
    childage=df['hw1'],
    vaccination=(df['h10'] == 1).astype(int),  # boolean to int conversion
    
    weight=df['hw2'] / 10,
    height=df['hw3'] / 10,
    hta=pd.to_numeric(df['hw70'], errors='coerce') / 100,
    wta=pd.to_numeric(df['hw71'], errors='coerce') / 100,
    wtht=pd.to_numeric(df['hw72'], errors='coerce') / 100,
    
    male=(df['b4'] == 'male').astype(int),
    prebrthint=df['b11'],
    brthwgt=pd.to_numeric(df['m19'], errors='coerce') / 100,
    urban=(df['v025'] == 'urban').astype(int),
    electricity=(df['v119'] == 'yes').astype(int),
    hlthcntr=(df['v394'] == 'yes').astype(int),
    notsmoking=(df['v463z'] == 'yes, does not use tobacco').astype(int),
    ethnicity=df['v131'],
    femalehh=(df['v151'] == 'female').astype(int),
    mariageage=df['v511'],
    wealth=df['v190'],
    # Wealth quintiles
    prst=(df['v190'] == 'poorest').astype(int),
    poorer=(df['v190'] == 'poorer').astype(int),
    mddl=(df['v190'] == 'middle').astype(int),
    rchr=(df['v190'] == 'richer').astype(int),
    rchst=(df['v190'] == 'richest').astype(int),
    #Indicators for born during and affected region
    bn = (df['b3'] >= 1266) & (df['b3'] <= 1271),
    ar = (df['v024'] == 'southern') | (df['v024'] == 'western'), 
    # generate indicator for those born in the affected regions (southern & western)
    sbreak = (df['hw1']>23)
)

df = df.assign(
    wasting=(df['wtht'] < -2).astype(int),  # wasting indicator
    underweight=(df['wta'] < -2).astype(int),  # underweight indicator
    stunting=(df['hta'] < -2).astype(int),  # stunting indicator
    # Education levels
    noeduc=(df['educlv'] == 'no education').astype(int),
    prmryeduc=(df['educlv'] == 'primary').astype(int),
    seceduc=(df['educlv'] == 'secondary').astype(int),
    hghreduc=(df['educlv'] == 'higher').astype(int),
    # Interaction terms
    bnar = df['bn'] & df['ar'],
    car=pd.to_numeric(df['childage'], errors='coerce') * df['ar'],
    cmale=pd.to_numeric(df['childage'], errors='coerce') * df['male'],
    cprebrthint=pd.to_numeric(df['childage'], errors='coerce') * df['prebrthint'],
    cchildnum=pd.to_numeric(df['childage'], errors='coerce') * pd.to_numeric(df['childnum'], errors='coerce'),
    cbreastfd=pd.to_numeric(df['childage'], errors='coerce') * pd.to_numeric(df['breastfd'], errors='coerce'),
    sbreak=(df['childage'] > 23).astype(int),
    sar=(df['childage'] > 23).astype(int) * df['ar'],
    lchildage=np.log(pd.to_numeric(df['childage'], errors='coerce')),
    lar=np.log(pd.to_numeric(df['childage'], errors='coerce')) * df['ar'],
    lmale=np.log(pd.to_numeric(df['childage'], errors='coerce')) * df['male'],
    cwealth=pd.to_numeric(df['childage'], errors='coerce') * pd.to_numeric(df['wealth'], errors='coerce'),
    swealth=pd.to_numeric(df['sbreak'], errors='coerce') * pd.to_numeric(df['wealth'], errors='coerce'),
    wgt=df['v005'] / 1000000
)

# Ensure the columns are in numeric format before performing comparisons
df = df.assign(
    v104=pd.to_numeric(df['v104'], errors='coerce'),
    hw13=pd.to_numeric(df['hw13'], errors='coerce'),
    b16=pd.to_numeric(df['b16'], errors='coerce'),
    v447=pd.to_numeric(df['v447'], errors='coerce')
)

# Now perform the filtering
df = df[(df['v104'] != 96)]
df = df[(df['b16'] != 0)]
# & (df['hw13'] <= 0) & (df['b16'] != 0) & (df['v447'] <= 0)]

# Remove NaN and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df_cleaned = df.dropna(subset=['childage'])
df_cleaned = df.dropna(subset=['hta'])
df_cleaned = df.dropna(subset=['wta'])


# Save to CSV
df_cleaned.to_csv('cleaned_data.csv', index=False)



