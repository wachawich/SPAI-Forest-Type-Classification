import pandas as pd
import pandas as pd
import numpy as np

train_df = pd.read_csv('train.csv')
train_df

b1 = train_df['b1']
b11 = train_df['b11']
b12 = train_df['b12']
b2 = train_df['b2']
b3 = train_df['b3']
b4 = train_df['b4']
b5 = train_df['b5']
b6 = train_df['b6']
b7 = train_df['b7']
b8 = train_df['b8']
b8a = train_df['b8_a']
b9 = train_df['b9']

train_df['NDVI'] = (train_df['b8'] - train_df['b4']) / (train_df['b8'] + train_df['b4'])
train_df['EVI'] = 2.5 * ((train_df['b8'] - train_df['b4']) / (train_df['b8'] + 6 * train_df['b4'] - 7.5 * train_df['b2'] + 1.01))
train_df['NDWI'] = (train_df['b3'] - train_df['b8']) / (train_df['b3'] + train_df['b8'])
train_df['SAVI'] = (train_df['b8'] - train_df['b4']) * (1 + 0.5) / (train_df['b8'] + train_df['b4'] + 0.5)
train_df['MSAVI'] = (2 * train_df['b8'] + 1 - ( (2 * train_df['b8'] + 1) ** 2 - 8 * (train_df['b8'] - train_df['b4'])) ** (1 / 2)) / 2
train_df['GNDVI'] = (train_df['b8'] - train_df['b3']) / (train_df['b8'] + train_df['b3'])
train_df['RENDVI'] = (train_df['b8'] - train_df['b5']) / (train_df['b8'] + train_df['b5'])
train_df['NDMI'] = (train_df['b8'] - train_df['b11']) / (train_df['b8'] + train_df['b11'])
train_df['GRVI'] = (train_df['b3'] - train_df['b4']) / (train_df['b3'] + train_df['b4'])
train_df['TVI'] = ( (train_df['b8'] - train_df['b4']) / (train_df['b8'] + train_df['b4'] + 0.5) ) ** (1 / 2)
train_df['MCARI'] = ((train_df['b5'] - train_df['b4']) - 0.2 * (train_df['b5'] - train_df['b3'])) / (train_df['b5'] / train_df['b4'])
train_df['BSI'] =  ((train_df['b11'] + train_df['b4']) - (train_df['b8'] + train_df['b2'])) / ((train_df['b11'] + train_df['b4']) + (train_df['b8'] + train_df['b2']))
train_df['NBR'] = (train_df['b8'] - train_df['b12']) / (train_df['b8'] + train_df['b12'])
train_df['MSI'] = train_df['b11'] / train_df['b8']

train_df['Adjusted transformed soil-adjusted VI'] = 1.22 * (b8 - 1.22 * b4 - 0.03) / (1.22 * b8 + b4 - 1.22 * 0.03 + 0.08 * (1 + 1.22 ** 2))
train_df['Aerosol free vegetation index 1600'] = b8 - 0.66 * b11 / (b8 + 0.66 * b11)
train_df['Aerosol free vegetation index 2100'] = b8 - 0.5 * b12 / (b8 + 0.56 * b12)
train_df['Alteration'] = b11 / b12
train_df['Anthocyanin reflectance index'] = 1 / b3 - 1 / b5
train_df['Atmospherically Resistant Vegetation Index 2'] = -0.18 + 1.17 * (b8 - b4) / (b8 + b4)
train_df['Blue-wide dynamic range vegetation index'] = (0.1 * b8 - b2) / (0.1 * b8 + b2)
train_df['Browning Reflectance Index'] = (1 / b3 - 1 / b5) / b8
train_df['Chlorophyll Absorption Ratio Index'] = (b5 / b4) * np.sqrt(((b5 - b3) / 150 * 670 + b4 + (b3 - ((b5 - b3) / 150 * 550))) ** 2) / np.sqrt(((b5 - b3) / (150 ** 2) + 1))
train_df['Chlorophyll Green'] = (b7 / b3) ** -1
train_df['Chlorophyll Index Green'] = (b8 / b3) - 1
train_df['Chlorophyll IndexRedEdge'] = (b8 / b5) - 1
train_df['Chlorophyll Red-Edge'] = (b7 / b5) ** -1
train_df['Chlorophyll vegetation index'] = b8 * b4 / (b3 ** 2)
train_df['Coloration Index'] = (b4 - b2) / b4
train_df['CRI550'] = (b2.astype(float) ** -1) - (b3.astype(float) ** -1)
train_df['CRI700'] = (b2.astype(float) ** -1) - (b5.astype(float) ** -1)
train_df['Datt1'] = (b8 - b5) / (b8 - b4)
train_df['Datt4'] = b4 / (b3 * b5)
train_df['Datt6'] = b8a / (b3 * b5)
train_df['Difference 678/500'] = b4 - b2
train_df['Difference 800/550'] = b8 - b3
train_df['Difference 800/680'] = b8 - b4
train_df['Difference 833/658'] = b8 - b4
train_df['Difference NIR/Green'] = b8 - b3
train_df['EVI'] = 2.5 * (b8 - b4) / (b8 + (6 * b4) + (-7.5 * b2) + 1)
train_df['EVI 2'] = 2.4 * (b8 - b4) / (b8 + b4 + 1)
train_df['Ferrous Silicates'] = b12 / b11
train_df['Global Environment Monitoring Index'] =  (2 * (b8 ** 2 - b4 ** 2) + 1.5 * b8 + 0.5 * b4) / (b8 + b4 + 0.5) * (1 - 0.25 * (2 * (b8 ** 2 - b4 ** 2) + 1.5 * b8 + 0.5 * b4) / (b8 + b4 + 0.5)) - ((b4 - 0.125) / (1 - b4))
train_df['Gossan'] = b11 / b4
train_df['Green atmospherically resistant vegetation index'] = (b8 - (b3 - (b2 - b4)))/(b8 - (b3 + (b2 - b4)))
train_df['Green leaf index'] = (2 * b3 - b4 - b2) / (2 * b3 + b4 + b2)
train_df['Green Normalized Difference Vegetation Index'] = (b8 - b3) / (b8 + b3)
train_df['Green Soil Adjusted Vegetation Index'] = (b8 - b3) / (b8 + b3 + 0.48) * (1 + 0.48)
train_df['Green-Blue NDVI'] = (b8 - (b3 + b2)) / (b8 + (b3 + b2))
train_df['Green-Red NDVI'] = (b8 - (b3 + b4)) / (b8 + (b3 + b4))
train_df['Hue'] = np.arctan((2 * b4 - b3 - b2) / (30.5 * (b3 - b2)))
train_df['Infrared percentage vegetation index'] = (b8 / (b8 + b4)) / 2 * (train_df['NDVI'] + 1)
train_df['Intensity'] = (1/30.5) * (b4 + b3 + b2)
train_df['Inverse reflectance 550'] = b3.astype(float) ** -1
train_df['Inverse reflectance 700'] = b5.astype(float) ** -1
train_df['Laterite'] = b11 / b12
train_df['Leaf Chlorophyll Index'] = (b8 - b5) / (b8 + b4)
train_df['Log Ratio'] = np.log(b8 / b4)
train_df['Maccioni'] = (b7 - b5) / (b7 - b4)
train_df['MCARI/MTVI2'] = (((b5 - b4) - 0.2 * (b5 - b3)) * (b5 / b4)) / (1.5 * (1.2 * (b8 - b3) - 2.5 * (b4 - b3)) / (np.sqrt((2 * b8 + 1) ** 2 - (6 * b8 - 5 * np.sqrt(4)) - 0.5)))
train_df['MCARI/OSAVI'] = ((b5 - b4) - 0.2 * (b5 - b3) * (b5 / b4))/ ((1 + 0.16) * (b8 - b4) / (b8 + b4 + 0.16))
train_df['mCRIG'] = (b2.astype(float) ** -1 - b3.astype(float) ** -1) * b8
train_df['mCRIRE'] = (b2.astype(float) ** -1 - b5.astype(float) ** -1) * b8
train_df['mND680'] = (b8 - b4) / (b8 + b4 - 2 * b1)
train_df['Modified anthocyanin reflectance index'] = (b3.astype(float) ** -1 - b5.astype(float) ** -1) * b8
train_df['Modified Chlorophyll Absorption in Reflectance Index'] = ((b5 - b4) - 0.2 * (b5 - b3)) * (b5 / b4)
train_df['Modified Chlorophyll Absorption in Reflectance Index 1'] = 1.2 * (2.5 * (b8 - b4) - 1.3 * (b8 - b3))
train_df['Modified Chlorophyll Absorption in Reflectance Index 2'] = 1.5 * (2.5 * (b8 - b4) - 1.3 * (b8 - b3)) / (np.sqrt((2 * b8 + 1) ** 2 - (6 * b8 - 5 * np.sqrt(b4)) - 0.5))
train_df['Modified NDVI'] = (b8 - b4) / (b8 + b4 - 2 * b1)
train_df['Modified Simple Ratio 670,800'] = ((b8 / b4) - 1) /np.sqrt((b8 - b4) + 1)
train_df['Modified Simple Ratio NIR/RED'] = ((b8 / b4) - 1) /np.sqrt((b8 / b4) + 1)
train_df['Modified Soil Adjusted Vegetation Index'] = (2 * b8 + 1 - np.sqrt((2 * b8 + 1) ** 2 - 8 * (b8 - b4)))/2
train_df['Modified Soil Adjusted Vegetation Index hyper'] = (0.5) * ((2 * b8 + 1) - np.sqrt((2 * b8 + 1) ** 2 - 8 * (b8 - b4)))
train_df['Modified Triangular Vegetation Index 1'] = 1.2 * (1.2 * (b8 - b3) - 2.5 * (b4 - b3))
train_df['Modified Triangular Vegetation Index 2'] = 1.5 * (1.2 * (b8 - b3) - 2.5 * (b4 - b3)) / np.sqrt((2 * b8 + 1) ** 2 - (6 * b8 - 5 * np.sqrt(b4)) - 0.5)
train_df['Norm G'] = b3 / (b8 + b4 + b3)
train_df['Norm NIR'] = b8 / (b8 + b4 + b3)
train_df['Norm R'] = b4 / (b8 + b4 + b3)
train_df['Normalized Difference 550/450'] = (b3 - b1) / (b3 + b1)
train_df['Normalized Difference 550/650'] = (b3 - b4) / (b3 + b4)
train_df['Normalized Difference 774/677'] = (b7 - b4) / (b7 + b4)
train_df['Normalized Difference 780/550'] = (b7 - b3) / (b7 + b3)
train_df['Normalized Difference 782/666'] = (b7 - b4) / (b7 + b4)
train_df['Normalized Difference 790/670'] = (b7 - b4) / (b7 + b4)
train_df['Normalized Difference 800/2170'] = (b8 - b12) / (b8 + b12)
train_df['Normalized Difference 800/470'] = (b8 - b2) / (b8 + b2)
train_df['Normalized Difference 800/500'] = (b8 - b2) / (b8 + b2)
train_df['Normalized Difference 800/550'] = (b8 - b3) / (b8 + b3)
train_df['Normalized Difference 800/650'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference 800/675'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference 800/680'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference 819/1600'] = (b8 - b11) / (b8 + b11)
train_df['Normalized Difference 819/1649'] = (b8 - b11) / (b8 + b11)
train_df['Normalized Difference 820/1600'] = (b8 - b11) / (b8 + b11)
train_df['Normalized Difference 827/668'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference 833/1649'] = (b8 - b11) / (b8 + b11)
train_df['Normalized Difference 833/658'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference 860/1640'] = (b8a - b11) / (b8a + b11)
train_df['Normalized Difference 895/675'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference Green/Red'] = (b3 - b4) / (b3 + b4)
train_df['Normalized Difference NIR/Blue'] = (b8 - b2) / (b8 + b2)
train_df['Normalized Difference NIR/Green'] = (b8 - b3) / (b8 + b3)
train_df['Normalized Difference NIR/Red'] = (b8 - b4) / (b8 + b4)
train_df['Normalized Difference NIR/Rededge'] = (b8 - b5) / (b8 + b5)
train_df['Normalized Difference Red/Green'] = (b4 - b3) / (b4 + b3)
train_df['Normalized Difference Salinity Index'] = (b11 - b12) / (b11 + b12)
train_df['Normalized Difference Vegetation Index 690-710'] = (b8 - b5) / (b8 + b5)
train_df['Optimized Soil Adjusted Vegetation Index'] = (1 + 0.16) * (b8 - b4) / (b8 + b4 + 0.16)
train_df['Pan NDVI'] = (b8 - (b3 + b4 + b2)) / (b8 + (b3 + b4 + b2))
train_df['RDVI'] = (b8 - b4) / ((b8 + b4).pow(0.5))
train_df['RDVI2'] = (b8 - b4) / ((b8 + b4).pow(0.5))
train_df['Red edge 1'] = b5 / b4
train_df['Red edge 2'] = (b5 - b4) / (b5 + b4)
train_df['Red-Blue NDVI'] = (b8 - (b4 + b2)) / (b8 + (b4 + b2))
train_df['Red-Edge Inflection Point 1'] = 700 + 40 * ((((b4 + b7) / 2) - b5) / (b6 - b5))
train_df['Red-Edge Inflection Point 2'] = 702 + 40 * ((((b4 + b7) / 2) - b5) / (b6 - b5))
train_df['Red-Edge Inflection Point 3'] = 705 + 35 * ((((b4 + b7) / 2) - b5) / (b6 - b5))
train_df['Red-Edge Position Linear Interpolation'] = 700 + 40 * ((((b4 + b7) / 2) - b5) / (b6 - b5))
train_df['Reflectance at the inflexion point'] = (b4 + b7) / 2
train_df['Renormalized Difference Vegetation Index'] = (b8 - b4) / ((b8 + b4).pow(0.5))
train_df['Simple Ratio 1600/820'] = b11 / b8
train_df['Simple Ratio 1650/2218'] = b11 / b12
train_df['Simple Ratio 440/740'] = b1 / b6
train_df['Simple Ratio 450/550'] = b1 / b3
train_df['Simple Ratio 520/670'] = b2 / b4
train_df['Simple Ratio 550/800'] = b3 / b8
train_df['Simple Ratio 560/658'] = b3 / b4
train_df['Simple Ratio 675/555'] = b4 / b3
train_df['Simple Ratio 675/705'] = b4 / b5
train_df['Simple Ratio 700'] = 1 / b5
train_df['Simple Ratio 710/670'] = b5 / b4
train_df['Simple Ratio 735/710'] = b6 / b5
train_df['Simple Ratio 774/677'] = b7 / b4
train_df['Simple Ratio 800/2170'] = b8 / b12
train_df['Simple Ratio 800/500'] = b8 / b2
train_df['Simple Ratio 810/560'] = b8 / b3
train_df['Simple Ratio 833/1649'] = b8 / b11
train_df['Simple Ratio 833/658'] = b8 / b4
train_df['Simple Ratio 850/710'] = b8 / b5
train_df['Simple Ratio 860/550'] = b8a / b3
train_df['Simple Ratio 860/708'] = b8a / b5
train_df['Simple Ratio NIR/700-715'] = b8 / b5
train_df['Simple Ratio NIR/G'] = b8 / b3
train_df['Simple Ratio NIR/RED'] = b8 / b4
train_df['Simple Ratio NIR/Rededge'] = b8 / b5
train_df['Simple Ratio Red/Blue'] = b4 / b2
train_df['Simple Ratio Red/Green'] = b4 / b3
train_df['Simple Ratio Red/NIR'] = b4 / b8
train_df['Soil Adjusted Vegetation Index'] = (b8 - b4) / (b8 + b4 + 0.48) * (1 + 0.48)
train_df['SQRT(IR/R)'] = (b8 / b4).pow(0.5)
train_df['Structure Intensive Pigment Index 1'] = (b8 - b1) / (b8 - b4)
train_df['Structure Intensive Pigment Index 3'] = (b8 - b2) / (b8 - b4)
train_df['TCARI/OSAVI'] = (3 * ((b5 - b4) - (0.2 * (b5 - b3) * (b5 / b4)))) / ((1 + 0.16) * ((b8 - b4)/(b8 + b4 + 0.16)))
train_df['Transformed Chlorophyll Absorbtion Ratio'] = 3 * ((b5 - b4) - (0.2 * (b5 - b3) * (b5 / b4)))
train_df['Transformed NDVI'] = ((b8 - b4) / (b8 + b4) + 0.5).pow(0.5)
train_df['Triangular chlorophyll index'] = 1.2 * (b5 - b3) - 1.5 * (b4 - b3) * (b5 / b4).pow(0.5)
train_df['Vegetation Index 700'] = (b5 - b4) / (b5 + b4)
train_df['Visible Atmospherically Resistant Index Green'] = (b3 - b4) / (b3 + b4 + b2)
train_df['Visible Atmospherically Resistant Indices RedEdge'] = (b5 - b4) / (b5 + b4)
train_df['Wide Dynamic Range Vegetation Index'] = (0.1 * b8 - b4) / (0.1 * b8 + b4)

feature = [
    'b1', 'b11', 'b12', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8_a', 'b9', 'NDVI', 'EVI', 'NDWI', 'SAVI', 'MSAVI',
    'GNDVI', 'RENDVI', 'NDMI', 'GRVI', 'TVI', 'MCARI', 'BSI', 'NBR', 'MSI', 'Adjusted transformed soil-adjusted VI',
    'Aerosol free vegetation index 1600', 'Aerosol free vegetation index 2100', 'Alteration', 'Anthocyanin reflectance index',
    'Atmospherically Resistant Vegetation Index 2', 'Blue-wide dynamic range vegetation index', 'Browning Reflectance Index',
    'Chlorophyll Absorption Ratio Index', 'Chlorophyll Green', 'Chlorophyll Index Green', 'Chlorophyll IndexRedEdge',
    'Chlorophyll Red-Edge', 'Chlorophyll vegetation index', 'Coloration Index', 'CRI550', 'CRI700', 'Datt1', 'Datt4',
    'Datt6', 'Difference 678/500', 'Difference 800/550', 'Difference 800/680', 'Difference 833/658', 'Difference NIR/Green',
    'EVI 2', 'Ferrous Silicates', 'Global Environment Monitoring Index', 'Gossan', 'Green atmospherically resistant vegetation index',
    'Green leaf index', 'Green Normalized Difference Vegetation Index', 'Green Soil Adjusted Vegetation Index', 'Green-Blue NDVI',
    'Green-Red NDVI', 'Hue', 'Infrared percentage vegetation index', 'Intensity', 'Inverse reflectance 550', 'Inverse reflectance 700',
    'Laterite', 'Leaf Chlorophyll Index', 'Log Ratio', 'Maccioni', 'MCARI/MTVI2', 'MCARI/OSAVI', 'mCRIG', 'mCRIRE', 'mND680',
    'Modified anthocyanin reflectance index', 'Modified Chlorophyll Absorption in Reflectance Index',
    'Modified Chlorophyll Absorption in Reflectance Index 1', 'Modified Chlorophyll Absorption in Reflectance Index 2',
    'Modified NDVI', 'Modified Simple Ratio 670,800', 'Modified Simple Ratio NIR/RED', 'Modified Soil Adjusted Vegetation Index',
    'Modified Soil Adjusted Vegetation Index hyper', 'Modified Triangular Vegetation Index 1', 'Modified Triangular Vegetation Index 2',
    'Norm G', 'Norm NIR', 'Norm R', 'Normalized Difference 550/450', 'Normalized Difference 550/650', 'Normalized Difference 774/677',
    'Normalized Difference 780/550', 'Normalized Difference 782/666', 'Normalized Difference 790/670', 'Normalized Difference 800/2170',
    'Normalized Difference 800/470', 'Normalized Difference 800/500', 'Normalized Difference 800/550', 'Normalized Difference 800/650',
    'Normalized Difference 800/675', 'Normalized Difference 800/680', 'Normalized Difference 819/1600', 'Normalized Difference 819/1649',
    'Normalized Difference 820/1600', 'Normalized Difference 827/668', 'Normalized Difference 833/1649', 'Normalized Difference 833/658',
    'Normalized Difference 860/1640', 'Normalized Difference 895/675', 'Normalized Difference Green/Red', 'Normalized Difference NIR/Blue',
    'Normalized Difference NIR/Green', 'Normalized Difference NIR/Red', 'Normalized Difference NIR/Rededge', 'Normalized Difference Red/Green',
    'Normalized Difference Salinity Index', 'Normalized Difference Vegetation Index 690-710', 'Optimized Soil Adjusted Vegetation Index',
    'Pan NDVI', 'RDVI', 'RDVI2', 'Red edge 1', 'Red edge 2', 'Red-Blue NDVI', 'Red-Edge Inflection Point 1', 'Red-Edge Inflection Point 2',
    'Red-Edge Inflection Point 3', 'Red-Edge Position Linear Interpolation', 'Reflectance at the inflexion point',
    'Renormalized Difference Vegetation Index', 'Simple Ratio 1600/820', 'Simple Ratio 1650/2218', 'Simple Ratio 440/740',
    'Simple Ratio 450/550', 'Simple Ratio 520/670', 'Simple Ratio 550/800', 'Simple Ratio 560/658', 'Simple Ratio 675/555',
    'Simple Ratio 675/705', 'Simple Ratio 700', 'Simple Ratio 710/670', 'Simple Ratio 735/710', 'Simple Ratio 774/677',
    'Simple Ratio 800/2170', 'Simple Ratio 800/500', 'Simple Ratio 810/560', 'Simple Ratio 833/1649', 'Simple Ratio 833/658',
    'Simple Ratio 850/710', 'Simple Ratio 860/550', 'Simple Ratio 860/708', 'Simple Ratio NIR/700-715', 'Simple Ratio NIR/G',
    'Simple Ratio NIR/RED', 'Simple Ratio NIR/Rededge', 'Simple Ratio Red/Blue', 'Simple Ratio Red/Green', 'Simple Ratio Red/NIR',
    'Soil Adjusted Vegetation Index', 'SQRT(IR/R)', 'Structure Intensive Pigment Index 1', 'Structure Intensive Pigment Index 3',
    'TCARI/OSAVI', 'Transformed Chlorophyll Absorbtion Ratio', 'Transformed NDVI', 'Triangular chlorophyll index', 'Vegetation Index 700',
    'Visible Atmospherically Resistant Index Green', 'Visible Atmospherically Resistant Indices RedEdge', 'Wide Dynamic Range Vegetation Index'
]

target = ['nforest_type']

X = train_df[feature]
y = train_df[target]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

#----------------------------------------------------------------------------------------

y_train_2 = y_train.copy()

num_list = []
for i in y_train_2['nforest_type']:
  if i == "MDF":
    num_list.append(0)
  elif i == "DDF":
    num_list.append(1)
  else:
    num_list.append(2)
y_train_2['nforest_type'] = num_list

y_val_2 = y_val.copy()

num_list = []
for i in y_val_2['nforest_type']:
  if i == "MDF":
    num_list.append(0)
  elif i == "DDF":
    num_list.append(1)
  else:
    num_list.append(2)
y_val_2['nforest_type'] = num_list

#----------------------------------------------------------------------------------------

import catboost as cb
from sklearn.metrics import mean_squared_error
import optuna

def objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-2, 0.1, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train_2)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val_2, predictions, squared=False)
    return rmse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)