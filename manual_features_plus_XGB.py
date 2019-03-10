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
from nptdms import TdmsFile

###################################################################
##            Define methods for feature engineering.            ##
###################################################################

# utility method for feature creation
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def create_features(row, segment, train_X, postfix=''):
    """
    postfix: String, is for when calling create_features more than once for feature creation, 
                     so we can name each new set differently

                     calling create_features(i, data, dataf, "mic1")
                     outputs mean_mic1, std_mic1 .... all the way to the last feature created
    """

    # pd.values returns a numpy representation of the data
    # pd.Series creates a one-dimensional ndarray with axis labels (including time series)
    x_ts = pd.Series(segment.values)
    zc = np.fft.fft(x_ts)
    
    train_X.loc[row, 'mean' + postfix] = x_ts.mean()
    train_X.loc[row, 'std' + postfix] = x_ts.std()
    train_X.loc[row, 'max' + postfix ] = x_ts.max()
    train_X.loc[row, 'min' + postfix] = x_ts.min()

    #FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    train_X.loc[row, 'Rmean' + postfix] = realFFT.mean()
    train_X.loc[row, 'Rstd' + postfix] = realFFT.std()
    train_X.loc[row, 'Rmax' + postfix] = realFFT.max()
    train_X.loc[row, 'Rmin' + postfix] = realFFT.min()
    train_X.loc[row, 'Imean' + postfix] = imagFFT.mean()
    train_X.loc[row, 'Istd' + postfix] = imagFFT.std()
    train_X.loc[row, 'Imax' + postfix] = imagFFT.max()
    train_X.loc[row, 'Imin' + postfix] = imagFFT.min()

    # train_X.loc[row, 'Rmean_last_5000' + postfix] = realFFT[-5000:].mean()
    # train_X.loc[row, 'Rstd__last_5000' + postfix] = realFFT[-5000:].std()
    # train_X.loc[row, 'Rmax_last_5000' + postfix] = realFFT[-5000:].max()
    # train_X.loc[row, 'Rmin_last_5000' + postfix] = realFFT[-5000:].min()
    # train_X.loc[row, 'Rmean_last_15000' + postfix] = realFFT[-15000:].mean()
    # train_X.loc[row, 'Rstd_last_15000' + postfix] = realFFT[-15000:].std()
    # train_X.loc[row, 'Rmax_last_15000' + postfix] = realFFT[-15000:].max()
    # train_X.loc[row, 'Rmin_last_15000' + postfix] = realFFT[-15000:].min()

    # # fft statistics "first"
    # train_X.loc[row, 'Rmean_first_5000' + postfix] = realFFT[:5000].mean()
    # train_X.loc[row, 'Rstd__first_5000' + postfix] = realFFT[:5000].std()
    # train_X.loc[row, 'Rmax_first_5000' + postfix] = realFFT[:5000].max()
    # train_X.loc[row, 'Rmin_first_5000' + postfix] = realFFT[:5000].min()
    # train_X.loc[row, 'Rmean_first_15000' + postfix] = realFFT[:15000].mean()
    # train_X.loc[row, 'Rstd_first_15000' + postfix] = realFFT[:15000].std()
    # train_X.loc[row, 'Rmax_first_15000' + postfix] = realFFT[:15000].max()
    # train_X.loc[row, 'Rmin_first_15000' + postfix] = realFFT[:15000].min()

    # librosa
    train_X.loc[row, 'MFCC_mean' + postfix] = librosa.feature.mfcc(np.float64(x_ts)).mean()
    train_X.loc[row, 'MFCC_std' + postfix] = librosa.feature.mfcc(np.float64(x_ts)).std()
    train_X.loc[row, 'MFCC_max' + postfix] = librosa.feature.mfcc(np.float64(x_ts)).max()
    train_X.loc[row, 'MFCC_min' + postfix] = librosa.feature.mfcc(np.float64(x_ts)).min()

    # train_X.loc[row, 'MFCC_mean_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].mean()
    # train_X.loc[row, 'MFCC_std_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].std()
    # train_X.loc[row, 'MFCC_max_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].max()
    # train_X.loc[row, 'MFCC_min_last15000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-15000:].min()

    # train_X.loc[row, 'MFCC_mean_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].mean()
    # train_X.loc[row, 'MFCC_std_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].std()
    # train_X.loc[row, 'MFCC_max_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].max()
    # train_X.loc[row, 'MFCC_min_last5000'] = librosa.feature.mfcc(np.float64(x_ts), sr=4000000)[-5000:].min()

    train_X.loc[row, 'mean_change_abs' + postfix] = np.mean(np.diff(x_ts))
    train_X.loc[row, 'mean_change_rate' + postfix] = np.mean(np.nonzero((np.diff(x_ts) / x_ts[:-1]))[0])
    train_X.loc[row, 'abs_max' + postfix] = np.abs(x_ts).max()
    train_X.loc[row, 'abs_min' + postfix] = np.abs(x_ts).min()
    
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
    
    train_X.loc[row, 'max_to_min' + postfix] = x_ts.max() / np.abs(x_ts.min())
    train_X.loc[row, 'max_to_min_diff' + postfix] = x_ts.max() - np.abs(x_ts.min())
    train_X.loc[row, 'count_big' + postfix] = len(x_ts[np.abs(x_ts) > 500])
    train_X.loc[row, 'sum' + postfix] = x_ts.sum()
    
    train_X.loc[row, 'q95' + postfix] = np.quantile(x_ts, 0.95)
    train_X.loc[row, 'q99' + postfix] = np.quantile(x_ts, 0.99)
    train_X.loc[row, 'q05' + postfix] = np.quantile(x_ts, 0.05)
    train_X.loc[row, 'q01' + postfix] = np.quantile(x_ts, 0.01)
    
    train_X.loc[row, 'abs_q95' + postfix] = np.quantile(np.abs(x_ts), 0.95)
    train_X.loc[row, 'abs_q99' + postfix] = np.quantile(np.abs(x_ts), 0.99)
    train_X.loc[row, 'abs_q05' + postfix] = np.quantile(np.abs(x_ts), 0.05)
    train_X.loc[row, 'abs_q01' + postfix] = np.quantile(np.abs(x_ts), 0.01)

    train_X.loc[row, 'mad' + postfix] = x_ts.mad()
    train_X.loc[row, 'kurt' + postfix] = x_ts.kurtosis()
    train_X.loc[row, 'skew' + postfix] = x_ts.skew()
    train_X.loc[row, 'med' + postfix] = x_ts.median()

    train_X.loc[row, 'Moving_average_700_mean' + postfix] = x_ts.rolling(window=700).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_1500_mean' + postfix] = x_ts.rolling(window=1500).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_3000_mean' + postfix] = x_ts.rolling(window=3000).mean().mean(skipna=True)
    train_X.loc[row, 'Moving_average_6000_mean' + postfix] = x_ts.rolling(window=6000).mean().mean(skipna=True)
    
    # Exponential MAVE
    ewma = pd.Series.ewm
    train_X.loc[row, 'exp_Moving_average_300_mean' + postfix] = (ewma(x_ts, span=300).mean()).mean(skipna=True)
    train_X.loc[row, 'exp_Moving_average_3000_mean' + postfix] = ewma(x_ts, span=3000).mean().mean(skipna=True)
    train_X.loc[row, 'exp_Moving_average_30000_mean' + postfix] = ewma(x_ts, span=30000).mean().mean(skipna=True)

    train_X.loc[row, 'iqr' + postfix] = np.subtract(*np.percentile(x_ts, [75, 25]))
    train_X.loc[row, 'q999' + postfix] = np.quantile(x_ts,0.999)
    train_X.loc[row, 'q001' + postfix] = np.quantile(x_ts,0.001)
    train_X.loc[row, 'ave10' + postfix] = stats.trim_mean(x_ts, 0.1)

    # trend features
    train_X.loc[row, 'trend' + postfix] = add_trend_feature(x_ts.values)
    train_X.loc[row, 'abs_trend' + postfix] = add_trend_feature(x_ts.values, abs_values=True)

    for windows in [10, 100, 1000]:
        x_roll_std = x_ts.rolling(windows).std().dropna().values
        x_roll_mean = x_ts.rolling(windows).mean().dropna().values
        
        train_X.loc[row, 'ave_roll_std_' + str(windows) + postfix] = x_roll_std.mean()
        train_X.loc[row, 'std_roll_std_' + str(windows) + postfix] = x_roll_std.std()
        train_X.loc[row, 'max_roll_std_' + str(windows) + postfix] = x_roll_std.max()
        train_X.loc[row, 'min_roll_std_' + str(windows) + postfix] = x_roll_std.min()
        train_X.loc[row, 'q01_roll_std_' + str(windows) + postfix] = np.quantile(x_roll_std, 0.01)
        train_X.loc[row, 'q05_roll_std_' + str(windows) + postfix] = np.quantile(x_roll_std, 0.05)
        train_X.loc[row, 'q95_roll_std_' + str(windows) + postfix] = np.quantile(x_roll_std, 0.95)
        train_X.loc[row, 'q99_roll_std_' + str(windows) + postfix] = np.quantile(x_roll_std, 0.99)
        train_X.loc[row, 'av_change_abs_roll_std_' + str(windows) + postfix] = np.mean(np.diff(x_roll_std))
        train_X.loc[row, 'av_change_rate_roll_std_' + str(windows) + postfix] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        train_X.loc[row, 'abs_max_roll_std_' + str(windows) + postfix] = np.abs(x_roll_std).max()
        
        train_X.loc[row, 'ave_roll_mean_' + str(windows) + postfix] = x_roll_mean.mean()
        train_X.loc[row, 'std_roll_mean_' + str(windows) + postfix] = x_roll_mean.std()
        train_X.loc[row, 'max_roll_mean_' + str(windows) + postfix] = x_roll_mean.max()
        train_X.loc[row, 'min_roll_mean_' + str(windows) + postfix] = x_roll_mean.min()
        train_X.loc[row, 'q01_roll_mean_' + str(windows) + postfix] = np.quantile(x_roll_mean, 0.01)
        train_X.loc[row, 'q05_roll_mean_' + str(windows) + postfix] = np.quantile(x_roll_mean, 0.05)
        train_X.loc[row, 'q95_roll_mean_' + str(windows) + postfix] = np.quantile(x_roll_mean, 0.95)
        train_X.loc[row, 'q99_roll_mean_' + str(windows) + postfix] = np.quantile(x_roll_mean, 0.99)
        train_X.loc[row, 'av_change_abs_roll_mean_' + str(windows) + postfix] = np.mean(np.diff(x_roll_mean))
        train_X.loc[row, 'av_change_rate_roll_mean_' + str(windows) + postfix] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        train_X.loc[row, 'abs_max_roll_mean_' + str(windows) + postfix] = np.abs(x_roll_mean).max()

###################################################################
##                 Transformation of tdms to csv.                ##
###################################################################

# read tdms files, and concatenate every two consecutive ones and save them as csv:

tdms_file_list =  glob.glob('../audio_tdms/*.tdms')
csv_file_list =  glob.glob('../audio_csv/*.csv')

# print(len(tdms_file_list))

mic_data = pd.DataFrame(columns=['mic1_motor','mic2'])
counter = 0
file_previous = None
#check if at least one csv file exists, if true, then skip creation of csv's
if os.path.isfile('../audio_csv/' + str(counter).zfill(2) + '-sound-mics.csv'):
    for i, file in enumerate(tqdm(tdms_file_list)):
        # add a condition to not go through this process over and over
        tdms_file = TdmsFile(file)
        tdms_df = tdms_file.as_dataframe().rename(index=str, columns={"/'Untitled'/'Micrófono 1 Motor'":"mic1_motor", "/'Untitled'/'Micrófono2'":"mic2"})
        #mic_data = pd.concat([mic_data, tdms_df], ignore_index=True)
        # check after first file
        if i > 0 and file_previous is not None:
            #print(file[23:-7])
            if file[23:-7]==file_previous:
                print("Here", i)
                mic_data = pd.concat([mic_data_previous, tdms_df], ignore_index=True)
                counter += 1
                # save then reset dataframe
                mic_data.to_csv('../audio_csv/' + str(counter).zfill(2) + '-sound-mics.csv', index=False)
                mic_data = pd.DataFrame(columns=['mic1_motor','mic2'])
                file_previous = None
            else:
                counter += 1
                #save previous to csv and update previous with new data
                mic_data_previous.to_csv('../audio_csv/' + str(counter).zfill(2) + '-sound-mics.csv', index=False)
                mic_data_previous = tdms_df
                file_previous = file[23:-7]
        # save first element in previous memory
        else:
            mic_data_previous = tdms_df
            file_previous = file[23:-7]
            # if this is the last file and it has no pair:
            if file_previous == tdms_file_list[len(tdms_file_list)-1]:
                mic_data_previous.to_csv('../audio_csv/' + str(126).zfill(2) + '-sound-mics.csv', index=False)
            
print("*** TDMS to CSV conversion finished ***")

# the last file was missing from the previous creation:
# tdms_file = TdmsFile(tdms_file_list[len(tdms_file_list)-1])
# tdms_df = tdms_file.as_dataframe().rename(index=str, columns={"/'Untitled'/'Micrófono 1 Motor'":"mic1_motor", "/'Untitled'/'Micrófono2'":"mic2"})
# tdms_df.to_csv('../audio_csv/' + str(126).zfill(2) + '-sound-mics.csv', index=False)


###################################################################
##                      Examine audio files                      ##
###################################################################

first_file = tdms_file_list[0]
start_time = first_file[23:-5]
#Look to sync data with the following time:
# print("Initial audio sample: ", start_time) # 07/12/2017 - 16:33:00

###################################################################
##      audio samples are not continuos, extract time stamps     ##
###################################################################

dates = []
for t in tdms_file_list:    
    # transform name into date:
    splitted = str.split(t[23:-5])
    # ignore date and seconds
    time = splitted[1:-1]
    # add fixed date
    time.extend(['00','07', '12', '2018']) #07/12/2017 ------Error in time stamp from source, it is actually 2018
    date_obj = datetime.datetime.strptime(' '.join(time), '%H %M %S %d %m %Y')
    date_string = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    dates.append(date_string)
    # remove duplicates
    dates = list(set(dates))
# print("Date: ", dates)
print("Unique sampled dates: ", len(dates))

###################################################################
##              Process audio_csv to create features             ##
###################################################################

# Create two different training sets, each 
features = pd.DataFrame()
features_mic1 = pd.DataFrame()
features_mic2 = pd.DataFrame()

csv_file_list =  glob.glob('../audio_csv/*.csv')
if not os.path.isfile('data_features.csv'):
    print("Feature creation process started...")
    for i, file in enumerate(tqdm(csv_file_list)):        
        audio_data = pd.read_csv(file)
        audio_data_mic1 = audio_data['mic1_motor']
        audio_data_mic2 = audio_data['mic2']
        create_features(i, audio_data_mic1, features_mic1, '_mic1')
        create_features(i, audio_data_mic2, features_mic2, '_mic2')
    features = pd.concat([features_mic1,features_mic2], axis=1)
    print("*** Feature creation finished ***")
    features.to_csv('data_features.csv')
else:
    print("*** Features already created ***")
    features = pd.read_csv('data_features.csv', index_col=0)
    print("Shape of features: ", features.shape)

###################################################################
##      Select process samples according to audio time stamps    ##
###################################################################

if not os.path.isfile('target.csv'):
    target = pd.read_excel('../scattered_variables/Impact Finder Data - For MPC - Dec-Part1.xlsx', sheet_name='Sheet1')
    # select important columns: time and kwH
    time_col = target[target.columns[0]]
    target_col = target[target.columns[6]]
    # concatenate them together
    target_ = pd.concat([time_col,target_col], ignore_index=True, axis=1)
    # convert the time_col to a string column
    target_[0] = target_[0].astype(str)
    # set first column as index column
    target_ = target_.set_index([0])
    # filter rows by passing in dates extracted from audio files samples
    target = target_.loc[dates,:]
    print(target.columns)
    target = target.rename(index=str, columns={1:'kWh'})

    target.to_csv('target.csv')
    print("Shape of target: ", target.shape)
else:
    target = pd.read_csv('target.csv', index_col = 0)
    print("Shape of target: ", target.shape)

###################################################################
##                 Prepare the data for training                 ##
###################################################################

data = pd.concat([features, target], axis= 1)
X = features
y = target

def scale(dataframe):
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled_dataframe = pd.DataFrame(scaler.transform(dataframe), columns=dataframe.columns)
    return scaled_dataframe

X_scaled = scale(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=80)

###################################################################
##                        Train the model                        ##
###################################################################

if not os.path.isfile('gbregressor_model.joblib'):
#if not os.path.isfile('gbregressor_model_optimized.joblib'):
    print("********************************")
    
    print("Training process has started ...")
    
    print("********************************")
    gbregressor = ensemble.GradientBoostingRegressor(loss='ls', n_estimators=1300, criterion="mae", 
                                                    learning_rate=0.01, n_iter_no_change= 500,
                                                    subsample=1, max_depth=6)
    score = cross_val_score(gbregressor, X_train, y_train, scoring='neg_mean_absolute_error', cv=10, n_jobs=5)
    #print("Score for cv: ", score)
    #########################################
    # Best params:  {'learning_rate': 0.01, 'loss': 'ls', 'max_depth': 6, 'n_estimators': 1300, 'subsample': 1}
    #########################################
    gbparams = [
        {
            'loss':['ls','lad','huber'],
            "n_estimators":[500, 1000, 1300],
            "learning_rate":[0.05, 0.01, 0.005],
            "subsample":[0.5, 1],
            "max_depth": [3, 4, 5, 6],
        }
                ]

    #gbregressor_optimized = GridSearchCV(gbregressor, gbparams, scoring='neg_mean_absolute_error', cv = 3, n_jobs = 5)
    #gbregressor_optimized.fit(X_train, y_train)
    #print("Best params: ", gbregressor_optimized.best_params_)
    #dump(gbregressor_optimized, 'gbregressor_model_optimized.joblib')
    gbregressor.fit(X_train, y_train)
    dump(gbregressor, 'gbregressor_model.joblib')
    print("The training process has finished")
else:
    #gbregressor_optimized = load('gbregressor_model_optimized.joblib')
    gbregressor = load('gbregressor_model.joblib')

# PREDICTION USING VALIDATION DATA
y_predict = gbregressor.predict(X_val)
#y_predict = gbregressor_optimized.predict(X_val)

mae_eval = metrics.mean_absolute_error(y_val, y_predict)

print("Mean Absolute Error(eval): " +  str(mae_eval))

title = "predict vs validation" 
fig, ax1 = plt.subplots(figsize=(12, 8))
plt.title(title)
plt.plot(y_val.values, color='r')
ax1.set_ylabel('y_validation', color='r')
plt.legend(['prediction'], loc=(0.01, 0.95))
plt.plot(y_predict, color='b')
ax2 = ax1.twinx()
ax2.set_ylabel('y_prediction', color='b')
plt.legend(['groundT'], loc=(0.01, 0.9))
plt.grid(True)
plt.show()