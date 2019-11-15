# -*- coding: utf-8 -*-
import pandas as pd
from RetrieveData import ObservedData
from collections import defaultdict

microHumidityData = "/home/decide/Data/Climato/Donnees_brutes/URBIO/Nantes/URBIO_Nantes_RH.csv"
microTemperatureData = "/home/decide/Data/Climato/Donnees_brutes/URBIO/Nantes/URBIO_Nantes_T.csv"

df_T = pd.read_csv(microTemperatureData, header = 0, index_col = 0, parse_dates = True)
df_RH = pd.read_csv(microHumidityData, header = 0, index_col = 0, parse_dates = True)

# Create the dictionary containing the year and month informations to recover
# meteorological data from the MeteoFrance archives
dic_time_range = createYearsAndMonthsDic(df_T)
# Load the Météo-France data from the local files
df_W_all = ObservedData.loadMeteoFranceSynop(cityCode = 7222,\
                         yearsAndMonths = dic_time_range)

var2keep = pd.Series(["Tair", "RHair", "WindSpeed", "WindDir", "Nebulosity", \
                      "Patmo", "etat_sol", "Precipitations"],\
                     index = ["t", "u", "ff", "dd", "n", "pres", "etat_sol", "rr24"])


# Recover only certain variables for the analysis and rename them
df_W = df_W_all[var2keep.index]
df_W.columns = var2keep[var2keep.index]

# Drop NaN values and potential duplicates (when importing by months...)
# and then reindex according to micro-meteorological indexes and interpolate
limit_interp = 12
df_W_wna = df_W.dropna(how = "all").drop_duplicates()
df_W_int = df_W_wna.reindex(df_T.index).interpolate(limit = limit_interp)

def createYearsAndMonthsDic(df):
    """ Create a dictionary of years and the corresponding months that are present
    in the micro-meteorological data.
    
        Pameters
		_ _ _ _ _ _ _ _ _ _ 
							
			df : pandas.DataFrame
				dataFrame containing the micro-meteorological data
            
        Returns
		_ _ _ _ _ _ _ _ _ _ 
							
        		A dictionary containing as key the years present in the data and
            as value a list of months"""
    t_start = pd.datetime(df.sort_index().index[0].year, 
                          df.sort_index().index[0].month, 1)
    t_end = pd.datetime(df.sort_index().index[-1].year, 
                        df.sort_index().index[-1].month, 1)

    # Create the datetime index containing each month of each year
    date_range = pd.date_range(start = t_start, end = t_end, freq = pd.offsets.MonthBegin(1))
    
    # Create a dictionary having years as keys and list of months as values
    result = defaultdict(list)
    for k, v in zip(date_range.year, date_range.month):
        result[k].append(v)
    
    return result
            
def meanDayCharac(df, robust = True, perc_values = 0.9):
    """ Reconstitutes a mean day and the variability (low and high value) 
    around this mean for each column of DataFrame.

        Pameters
		_ _ _ _ _ _ _ _ _ _ 
							
			df : pandas.DataFrame
				dataFrame containing the data to average
			robust : boolean, default True
				if robust is True: calculates for each time step of a day Ti:
                    - df_mean = median(df(day1), df(day2), ...)
                    - df_high = quantile_high(df(day1), df(day2), ...)
                    - df_low = quantile_low(df(day1), df(day2), ...)
                if robust is False: calculates for each time step of a day Ti:
                    - df_mean = mean(df(day 1), df(day2), ...)
                    - df_high = mean_df + std(df(day 1), df(day2), ...)
                    - df_low = mean_df - std(df(day 1), df(day2), ...)
            perc_values : float, default 0.9
                Only if robust = True, percentage of the values to integrate
                within the range [low value, high value]
            
        Returns
		_ _ _ _ _ _ _ _ _ _ 
							
        		Three dataframes : the first being the mean day of df, the second
            the low values at this time of the day, the last the high values at
            this time of the day"""
    if robust == True:
        result_mean = df.groupby([df.index.hour, df.index.minute]).median()
        result_high = df.groupby([df.index.hour, df.index.minute]).quantile(1-(1-perc_values)/2)
        result_low = df.groupby([df.index.hour, df.index.minute]).quantile((1-perc_values)/2)
    else:
        result_mean = df.groupby([df.index.hour, df.index.minute]).mean()
        result_std = df.groupby([df.index.hour, df.index.minute]).std()
        result_high = result_mean + result_std
        result_low = result_mean - result_std
    
    # The resulting tables having MultiIndex (hour and minute), we need to
    # reindex using datetimes
    date_init = df.index[0]
    new_index = [pd.datetime(date_init.year, date_init.month, date_init.day, 
                             result_mean.index.codes[0][i], result_mean.index.codes[1][i]) 
                for i in range(0, result_mean.index.codes[0].size)]
    result_mean.index = new_index
    result_high.index = new_index
    result_low.index = new_index
    
    return result_mean, result_high, result_low

def filterTimeAndAverage(df_micro, df_W, filt_micro, filt_W, robust = False,
                         time_norm = False):
    """ Acts differently on the two input types:
            - for micro-meteorological data, filters a certain range of
            the day time (e.g.: ["8:15", "10:15"] or ["23:00", "01:00"])
            and average it to have a single daily value.
            - for each meteorological variable (wind speed, nebulosity, etc.), 
            filters a specific range of hours before the micro-meteorological
            filtering (e.g.: {"Tair": [-2, 2], "RHair": [-3, 2], 
                              "WindSpeed": [0, 2], 
                              "Nebulosity": [-10, -2],...})

        Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
			df_micro : pandas.DataFrame
				dataFrame containing the micro-meteorological data
            df_W : pandas.DataFrame
				dataFrame containing the meteorological data
			filt_micro : list of string
                list containing the begining and the end of the day time range
                chosen for averaging the micro-meteorological conditions
            filt_W : dictionary
                dictionary containing as key a meteorological variable (such as
                wind speed, air temperature, nebulosity, etc.) and as value
                a list containing the starting hour where the meteorological
                averaging should starts for this variable
                (0 is the time where the filt_micro starts, -5 is 5 hours
                before the time where the filt_micro starts, 2 is 2 hours
                after the time where the filt_micro starts), and the ending
                hour where the meteorological averaging should stop for this
                variable.
            robust : boolean, default False
                if robust = True, the median of the value recorded within the
                time_range is calculated, else the mean is calculated
            time_norm : boolean, default False
                Whether or not the time is normalized in order to have day range
                between 0 and 1 and night range between 1 and 2.
            
        Returns
		_ _ _ _ _ _ _ _ _ _ 
							
        		Two dataframes : the first being the mean micro-meteorological values
                during the given interval for each day, the second the mean
                meteorological condition for each meteorological interval (and for
                each day)"""
    # Filters the micro-meterological data
    if filt_micro[0] > filt_micro[1]:
        df_micro_filt = \
            df_micro[(df_micro.index.time >= pd.Timestamp(filt_micro[0]).time())+\
                     (df_micro.index.time < pd.Timestamp(filt_micro[1]).time())]  
    else:
        df_micro_filt = \
            df_micro[(df_micro.index.time >= pd.Timestamp(filt_micro[0]).time())*\
                     (df_micro.index.time < pd.Timestamp(filt_micro[1]).time())]
    # Average the interval for each day (starting the average at the beginning of
    # the interval)
    if robust == True:
        df_micro_result = df_micro_filt.resample("24H").median()
    else:
        df_micro_result = df_micro_filt.resample("24H").mean()
    
    # Filters and calculates the average for meteorological conditions
    df_W_result = pd.DataFrame(index = df_micro_result.index)
    for v in filt_W.keys():
        if robust == True:
            # Use moving average in order to "median" the values of each meteorological
            # variable according to the informations coming from "filt_W"
            df_W_avg = df_W[v].rolling(pd.offsets.Hour(filt_W[v][1]-filt_W[v][0])).median()
        else:
            # Use moving average in order to "mean" the values of each meteorological
            # variable according to the informations coming from "filt_W"
            df_W_avg = df_W[v].rolling(pd.offsets.Hour(filt_W[v][1]-filt_W[v][0])).mean()
    
        t_end_range = (pd.Timestamp(filt_micro[0]) + pd.offsets.Hour(filt_W[v][1])).time()
        df_W_filt = df_W_avg.at_time(t_end_range)
        # In order to attribute meteorological values to the right studied day
        # it is necessary to offset df_W values under certain conditions
        micro_start_hour = int(filt_micro[0].split(":")[0])
        # the end of df_W average is in the previous day
        if micro_start_hour < -filt_W[v][1]:
            df_W_filt.index = df_W_filt.index.date + pd.offsets.Day(1)
        # the end of df_W average is in the next day
        elif 24-micro_start_hour < filt_W[v][1]:
            df_W_filt.index = df_W_filt.index.date + pd.offsets.Day(-1)
        else:
            df_W_filt.index = df_W_filt.index.date
        df_W_result = df_W_result.join(df_W_filt, how = "outer")

    return df_micro_result, df_W_result