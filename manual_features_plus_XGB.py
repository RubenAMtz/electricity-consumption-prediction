import os
import glob
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
#from multiprocessing import Pool
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from joblib import load, dump
import time
import datetime
import librosa

###################################################################
## Transformation of tdms to csv implemented with matlab scripts.##
###################################################################

tdms_file_list =  glob.glob('../audio_tdms/*.tdms')
# First file with time stamp
first_file = tdms_file_list[0]
start_time = first_file[23:-5]
#Look to sync data with the following time:
print("Initial audio sample: ", start_time) # 07/12/2017 - 16:33:00

###################################################################
## Read transformed audio data and apply feature engineering.    ##
###################################################################

# utility method for feature creation
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def create_features(row, segment, train_X):
    # pd.values returns a numpy representation of the data
    # pd.Series creates a one-dimensional ndarray with axis labels (including time series)
    x_ts = pd.Series(segment.values)
    zc = np.fft.fft(x_ts)
    
    train_X.loc[row, 'mean'] = x_ts.mean()
    train_X.loc[row, 'std'] = x_ts.std()
    train_X.loc[row, 'max'] = x_ts.max()
    train_X.loc[row, 'min'] = x_ts.min()

    #FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    train_X.loc[row, 'Rmean'] = realFFT.mean()
    train_X.loc[row, 'Rstd'] = realFFT.std()
    train_X.loc[row, 'Rmax'] = realFFT.max()
    train_X.loc[row, 'Rmin'] = realFFT.min()
    train_X.loc[row, 'Imean'] = imagFFT.mean()
    train_X.loc[row, 'Istd'] = imagFFT.std()
    train_X.loc[row, 'Imax'] = imagFFT.max()
    train_X.loc[row, 'Imin'] = imagFFT.min()

    # train_X.loc[row, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    # train_X.loc[row, 'Rstd__last_5000'] = realFFT[-5000:].std()
    # train_X.loc[row, 'Rmax_last_5000'] = realFFT[-5000:].max()
    # train_X.loc[row, 'Rmin_last_5000'] = realFFT[-5000:].min()
    # train_X.loc[row, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    # train_X.loc[row, 'Rstd_last_15000'] = realFFT[-15000:].std()
    # train_X.loc[row, 'Rmax_last_15000'] = realFFT[-15000:].max()
    # train_X.loc[row, 'Rmin_last_15000'] = realFFT[-15000:].min()

    # # fft statistics "first"
    # train_X.loc[row, 'Rmean_first_5000'] = realFFT[:5000].mean()
    # train_X.loc[row, 'Rstd__first_5000'] = realFFT[:5000].std()
    # train_X.loc[row, 'Rmax_first_5000'] = realFFT[:5000].max()
    # train_X.loc[row, 'Rmin_first_5000'] = realFFT[:5000].min()
    # train_X.loc[row, 'Rmean_first_15000'] = realFFT[:15000].mean()
    # train_X.loc[row, 'Rstd_first_15000'] = realFFT[:15000].std()
    # train_X.loc[row, 'Rmax_first_15000'] = realFFT[:15000].max()
    # train_X.loc[row, 'Rmin_first_15000'] = realFFT[:15000].min()

    # librosa
    train_X.loc[row, 'MFCC_mean'] = librosa.feature.mfcc(np.float64(x_ts)).mean()
    train_X.loc[row, 'MFCC_std'] = librosa.feature.mfcc(np.float64(x_ts)).std()
    train_X.loc[row, 'MFCC_max'] = librosa.feature.mfcc(np.float64(x_ts)).max()
    train_X.loc[row, 'MFCC_min'] = librosa.feature.mfcc(np.float64(x_ts)).min()

    # train_X.loc[row, 'MFCC_mean_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].mean()
    # train_X.loc[row, 'MFCC_std_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].std()
    # train_X.loc[row, 'MFCC_max_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].max()
    # train_X.loc[row, 'MFCC_min_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].min()

    # train_X.loc[row, 'MFCC_mean_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].mean()
    # train_X.loc[row, 'MFCC_std_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].std()
    # train_X.loc[row, 'MFCC_max_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].max()
    # train_X.loc[row, 'MFCC_min_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].min()

    train_X.loc[row, 'mean_change_abs'] = np.mean(np.diff(x_ts))
    train_X.loc[row, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x_ts) / x_ts[:-1]))[0])
    train_X.loc[row, 'abs_max'] = np.abs(x_ts).max()
    train_X.loc[row, 'abs_min'] = np.abs(x_ts).min()
    
    # train_X.loc[row, 'std_first_50000'] = x_ts[:50000].std()
    # train_X.loc[row, 'std_last_50000'] = x_ts[-50000:].std()
    # train_X.loc[row, 'std_first_10000'] = x_ts[:10000].std()
    # train_X.loc[row, 'std_last_10000'] = x_ts[-10000:].std()
    
    # train_X.loc[row, 'avg_first_50000'] = x_ts[:50000].mean()
    # train_X.loc[row, 'avg_last_50000'] = x_ts[-50000:].mean()
    # train_X.loc[row, 'avg_first_10000'] = x_ts[:10000].mean()
    # train_X.loc[row, 'avg_last_10000'] = x_ts[-10000:].mean()
    
    # train_X.loc[row, 'min_first_50000'] = x_ts[:50000].min()
    # train_X.loc[row, 'min_last_50000'] = x_ts[-50000:].min()
    # train_X.loc[row, 'min_first_10000'] = x_ts[:10000].min()
    # train_X.loc[row, 'min_last_10000'] = x_ts[-10000:].min()
    
    # train_X.loc[row, 'max_first_50000'] = x_ts[:50000].max()
    # train_X.loc[row, 'max_last_50000'] = x_ts[-50000:].max()
    # train_X.loc[row, 'max_first_10000'] = x_ts[:10000].max()
    # train_X.loc[row, 'max_last_10000'] = x_ts[-10000:].max()
    
    train_X.loc[row, 'max_to_min'] = x_ts.max() / np.abs(x_ts.min())
    train_X.loc[row, 'max_to_min_diff'] = x_ts.max() - np.abs(x_ts.min())
    train_X.loc[row, 'count_big'] = len(x_ts[np.abs(x_ts) > 500])
    train_X.loc[row, 'sum'] = x_ts.sum()
    
    train_X.loc[row, 'q95'] = np.quantile(x_ts, 0.95)
    train_X.loc[row, 'q99'] = np.quantile(x_ts, 0.99)
    train_X.loc[row, 'q05'] = np.quantile(x_ts, 0.05)
    train_X.loc[row, 'q01'] = np.quantile(x_ts, 0.01)
    
    train_X.loc[row, 'abs_q95'] = np.quantile(np.abs(x_ts), 0.95)
    train_X.loc[row, 'abs_q99'] = np.quantile(np.abs(x_ts), 0.99)
    train_X.loc[row, 'abs_q05'] = np.quantile(np.abs(x_ts), 0.05)
    train_X.loc[row, 'abs_q01'] = np.quantile(np.abs(x_ts), 0.01)

    train_X.loc[row, 'mad'] = x_ts.mad()
    train_X.loc[row, 'kurt'] = x_ts.kurtosis()
    train_X.loc[row, 'skew'] = x_ts.skew()
    train_X.loc[row, 'med'] = x_ts.median()

    train_X.loc[row, 'Moving_average_700_mean'] = x_ts.rolling(window=700).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_1500_mean'] = x_ts.rolling(window=1500).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_3000_mean'] = x_ts.rolling(window=3000).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_6000_mean'] = x_ts.rolling(window=6000).mean().mean(skipna=True)
    
    # Exponential MAVE
    ewma = pd.Series.ewm
    train_X.loc[row, 'exp_Moving_average_300_mean'] = (ewma(x_ts, span=300).mean()).mean(skipna=True)
    train_X.loc[row, 'exp_Moving_average_3000_mean'] = ewma(x_ts, span=3000).mean().mean(skipna=True)
    train_X.loc[row, 'exp_Moving_average_30000_mean'] = ewma(x_ts, span=30000).mean().mean(skipna=True)

    train_X.loc[row, 'iqr'] = np.subtract(*np.percentile(x_ts, [75, 25]))
    train_X.loc[row, 'q999'] = np.quantile(x_ts,0.999)
    train_X.loc[row, 'q001'] = np.quantile(x_ts,0.001)
    train_X.loc[row, 'ave10'] = stats.trim_mean(x_ts, 0.1)

    # trend features
    train_X.loc[row, 'trend'] = add_trend_feature(x_ts.values)
    train_X.loc[row, 'abs_trend'] = add_trend_feature(x_ts.values, abs_values=True)

    for windows in [10, 100, 1000]:
        x_roll_std = x_ts.rolling(windows).std().dropna().values
        x_roll_mean = x_ts.rolling(windows).mean().dropna().values
        
        train_X.loc[row, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        train_X.loc[row, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        train_X.loc[row, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        train_X.loc[row, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        train_X.loc[row, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        train_X.loc[row, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        train_X.loc[row, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        train_X.loc[row, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        train_X.loc[row, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        train_X.loc[row, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        train_X.loc[row, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        train_X.loc[row, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        train_X.loc[row, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        train_X.loc[row, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        train_X.loc[row, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        train_X.loc[row, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        train_X.loc[row, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        train_X.loc[row, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        train_X.loc[row, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        train_X.loc[row, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        train_X.loc[row, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        train_X.loc[row, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()



# Data to save the processed data:
train_X = pd.DataFrame()

csv_file_list =  glob.glob('../audio_csv/*.csv')
# if not os.path.isfile('train_X.csv'):
#     for i, file in enumerate(tqdm(csv_file_list)):
        
#         audio_data = pd.read_csv(file, header = None)
#         audio_data_scol = audio_data[0].append(audio_data[1]).reset_index(drop=True)
#         #create_features(i, audio_data_scol, train_X)
#         audio_data_scol = pd.DataFrame()

audio_data = pd.read_excel(csv_file_list[19])
print(audio_data.columns)
print(audio_data.shape)
print(audio_data.head())
audio_data_scol = audio_data[0].append(audio_data[1]).reset_index(drop=True)
    # train_X.to_csv('train_X.csv', index=False)
    


