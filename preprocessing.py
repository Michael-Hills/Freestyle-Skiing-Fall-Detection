import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
import numpy as np


def readRawData(foldername,labelled=True):
    """ Function to read the fall data csv files"""

    #Get file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(script_dir, foldername)
    all_files = glob.glob(os.path.join(filePath , "*.csv"))

    all_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    li = []


    #Columns to collect
    if labelled == True:
        skiprows = 0
    
        columns = ['Time,s','Pelvis Accel Sensor X,mG','Pelvis Accel Sensor Y,mG','Pelvis Accel Sensor Z,mG',
                'Pelvis Rot X,','Pelvis Rot Y,','Pelvis Rot Z,','Markers','MarkerNames','Fall']
        
    else:
        columns = ['Time,s','Pelvis Accel Sensor X,mG','Pelvis Accel Sensor Y,mG','Pelvis Accel Sensor Z,mG',
                'Pelvis Rot X,','Pelvis Rot Y,','Pelvis Rot Z,','Markers','MarkerNames']
        skiprows = 4

    
    columns2 = ['Time,s','Pelvis Accel Sensor X,mG','Pelvis Accel Sensor Y,mG','Pelvis Accel Sensor Z,mG',
               'Pelvis Rot X,','Pelvis Rot Y,','Pelvis Rot Z,']
    
    #Iterate through files
    for filename in all_files:
        try:        
            df = pd.read_csv(filename,skiprows=skiprows,usecols=columns,header=0,dtype={'MarkerNames': str})
            li.append(df)
        except:
            df = pd.read_csv(filename,skiprows=skiprows,usecols=columns2,header=0)
            li.append(df)    

    return li


def labelFalls():
    """Function to label fall regions in the dataset"""

    data = readRawData('SDSUfalldata',labelled=False)

    name = "Sample"
    count = 1

    for df in data:
        
        try:
        
            fallValues = np.zeros(len(df))
            temp = df[df['MarkerNames'].notna()]['MarkerNames'].str.contains('Fall')
            vals = list(temp[temp.values].index)
            
            for i in range(0,len(vals),2):
                fallValues[vals[i]:vals[i+1]] = 1
        except(KeyError):
            fallValues = np.zeros(len(df))
            
        
        df['Fall'] = fallValues
        df.to_csv(name+" "+str(count)+'.csv')
        count += 1
    
    

def conventionalSlidingWindow(data,window_size,step,lowPass=False):
    """Split the samples into windows of size window_size"""
  

    fall_count = 0
    nonFall_count = 0
    columns = ['Pelvis Accel Sensor X,mG', 'Pelvis Accel Sensor Y,mG',
       'Pelvis Accel Sensor Z,mG', 'Pelvis Rot X,', 'Pelvis Rot Y,',    
       'Pelvis Rot Z,']

    # iterate through each sample
    for df in data:
    
        n_examples = len(df)
        k=0


        # low pass filter if true
        if lowPass == True:
            x = df[columns].values.T
            filtX = lowPassFilter(x)
            df[columns] = filtX.T


        # split each sample into size window_size until end
        while(k * step + window_size < n_examples):

            temp = df.loc[k * step:k * step + window_size]
            try:
                if (temp['Fall'].value_counts()[1.0] > 0.05 * window_size):
                    fall_count += 1
                    temp.to_csv(os.getcwd() + "/4S/RawFall/sample" +str(fall_count) +'.csv')
                else:
                    nonFall_count += 1
                    temp.to_csv(os.getcwd() + "/4S/RawNonFall/sample" +str(nonFall_count) +'.csv')
            
            except:
                nonFall_count += 1
                temp.to_csv(os.getcwd() + "/4S/RawNonFall/sample" +str(nonFall_count) +'.csv')

            k += 1



def normalisedConventionalSlidingWindow(data,window_size,step,lowPass=False):
    """Normalise and split the samples into windows of size window_size"""
  

    fall_count = 0
    nonFall_count = 0

    columns = ['Pelvis Accel Sensor X,mG', 'Pelvis Accel Sensor Y,mG',
       'Pelvis Accel Sensor Z,mG', 'Pelvis Rot X,', 'Pelvis Rot Y,',    
       'Pelvis Rot Z,']

    conc = []

    # iterate through each sample
    for df in data:
        conc.append(df)

    alldata = pd.concat(conc,ignore_index=True)

    # low pass filter if true
    if lowPass == True:
        x = alldata[columns].values.T
        filtX = lowPassFilter(x)
        alldata[columns] = filtX.T
    

    # normalise the data
    scaler = MinMaxScaler()
    alldata[columns] = scaler.fit_transform(alldata[columns])

    
    
    n_examples = len(alldata)
    k = 0


    # split into windows until end
    while(k * step + window_size < n_examples):

        temp = alldata.loc[k * step:k * step + window_size]
        try:
            if (temp['Fall'].value_counts()[1.0] > 0.05 * window_size):
                fall_count += 1
                temp.to_csv(os.getcwd() + "/4S/RawFall/sample" +str(fall_count) +'.csv')
            else:
                nonFall_count += 1
                temp.to_csv(os.getcwd() + "/4S/RawNonFall/sample" +str(nonFall_count) +'.csv')
            
        except:
            nonFall_count += 1
            temp.to_csv(os.getcwd() + "/4S/RawNonFall/sample" +str(nonFall_count) +'.csv')
        
        k += 1

    

def lowPassFilter(data):

    """ Function to low pass filter the samples """


    b, a = butter(1, 0.1, btype='low')
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal







