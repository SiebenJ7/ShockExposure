import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat 

dtafile = '/Users/JuliusSiebenaller/Documents/HSG/BVWL/BA/BA_JS/ZMKR51FL.dta'
df = pd.read_stata(dtafile)
df.tail()