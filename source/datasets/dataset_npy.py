import os, glob, re
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import json
from scipy import stats

MODIS_SCALE_FACTOR = 1e-4
MODIS_FEATURES_IDX = [i for i in range(0, 7)]

class YieldDataset(data.Dataset):
    
    def __init__(self, image_folder, label_path, crop_type, return_id=False):
        super(YieldDataset, self).__init__()

        self.image_folder = image_folder
        self.label_path = label_path
        self.crop_type = crop_type
        self.return_id = return_id

        # read images
        self.image_paths = glob.glob(self.image_folder +'/**/*.npy',  recursive=True)
        file_names = [x.split('/')[-1].replace('.npy', '') for x in self.image_paths]
        years = np.array([int(x[:4]) for x in file_names])
        geoids = np.array([x.split('_')[-1] for x in file_names])
        df_image = pd.DataFrame({'id':geoids, 'year':years})

        self.df_image = df_image

        # read the labels
        with open(label_path) as f:
            yield_per_parcel = json.load(f)

        # print(yield_per_parcel)

        years, yields, ids = [], [], []
        for item in yield_per_parcel.items():
            years.append(int(item[0].split('_')[1]))
            ids.append(item[0].split('_')[0])
            yields.append(item[1])

        # print(years, yields, ids)

        df_yield = pd.DataFrame({'id':ids, 'year':years, 'yield':yields})
        # df_yield = df_yield[df_yield['year'] == year]
        # keep only intersection
        self.data = pd.merge(df_yield, df_image, how='inner')
        self.len = len(self.data.index)
        self.df_yield = df_yield

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        row = self.data.iloc[item]

        features = np.load(self.image_folder + '/{}_{}.npy'.format(row['year'], row['id']))

        #scale the MODIS bands based on the scale provided in the data documentation
        for i in MODIS_FEATURES_IDX:
            features[:,i] = features[:,i] * MODIS_SCALE_FACTOR

        if self.return_id:
            return torch.from_numpy(features), torch.tensor(row['yield']), row['id']
        else:
            return torch.from_numpy(features), torch.tensor(row['yield'])

    def get_county_data(self, county_id):
        df_entries = self.data[self.data['id'] == str(county_id)].copy()
        df_entries.sort_values(by='year', inplace=True)
        # print(df_entries)
        full_data = []
        for i, year in zip(df_entries.index, df_entries['year']):
          data = self.__getitem__(i)
          year_data = data[0].numpy()
          yield_data = data[1].numpy()

          year_arr = [year for _ in range(len(year_data))]
          yeild_arr = [yield_data for _ in range(len(year_data))]
          county_arr = [county_id for _ in range(len(year_data))]

          # year_data['year'] = year
          full_data.append(np.c_[county_arr, year_arr, yeild_arr, year_data])
          # full_data.append(np.c_[, year_data])
        
        result = np.concatenate(full_data)
        feature_names = ['county', 'year', 'yield', "RED", "NIR", "BLUE", "GREEN", "NIR2", "SWIR1", "SWIR2", "TEMP_MIN", "TEMP_MAX", "PRCP", "HEATWAVE INDEX", "DROUGHT INDEX", "NDVI", "EVI", "NDWI"]

        df =  pd.DataFrame(result, columns=feature_names)
        # df = self.remove_outliers_satellite(df)
        return df

    def remove_outliers_satellite(self, df):
        # removing outliers from satellite data
        cols = ["RED", "NIR", "BLUE", "GREEN", "NIR2", "SWIR1", "SWIR2"] # one or more
        Q1 = df[cols].quantile(0.01)
        Q3 = df[cols].quantile(0.99)
        IQR = Q3 - Q1
        df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
        return df
      