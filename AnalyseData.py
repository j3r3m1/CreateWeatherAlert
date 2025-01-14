# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:43:51 2020

@author: Jérémy Bernard
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from skyfield import api
from skyfield import almanac
import matplotlib.pylab as plt
from sklearn import tree
import graphviz
from math import ceil
import random
from joblib import dump
import pydot
import os

# For each key (corresponding to the month number), the associated season is given
SeasonDict={1: "Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10:"Autumn", 11:"Autumn", 12:"Winter"}

def GroupMonthBySeas(x):
    """Split data into seasons"""
    return SeasonDict[x.month]

def periodsIdentification(df2study, df_sun_events, path2SaveFig, 
                          seasonal_sorting = ["ALL", "SEASON"], 
                          dimensionlessDay = True, perc_values = 0.5, 
                          save = True, same_ax = False):
    """Reconstitutes a mean day and the variability (low and high value) 
    around this mean for each column of the DataFrame to study. This is done
    for the whole dataset or for each season, using real days or dimensionless
    days (cf. Bernard et al. 2017 or Oke ...). The resulting days are
    saved in figures which may be used to identify potential periods of interest.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df2study : DataFrame
				The micro-meteorological data to process
            df_sun_events : DataFrame
                The 'sunrise' and 'sunset' hours for each day of the 'df2study' dataset
            seasonal_sorting : List of String, default ["ALL", "SEASON"]
                A list of the seasonal sorting to apply. May contain :
                    -> "ALL" : no sorting is done, the day is representative of the whole dataset
                    -> "SEASON": one average day for each season
                    -> "MONTH" : one average day for each month
            dimensionlessDay : boolean, default True
                Whether or not the average days should also be plotted using a dimensionless scale
            same_ax : Boolean, default False
                Whether or not the curves are plotted on a same axis
            path2SaveFig : string
                Name of the URL where to save the resulting Figures if 'save' = True
			perc_values : float, default 0.5
				Percentage of the values that should be used for the low and high
                values of the variability
            save : boolean, default True
                Whether or not the figures should be saved

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			Return None"""
    # MEAN DAY FOR ALL DATA (AND WHOLE YEAR)
    # calculate a mean day and its variability around the mean and plot it
    if (seasonal_sorting.count("ALL")>0):
        df_mean, df_high, df_low = meanDayCharac(df2study, perc_values = perc_values)
        fig, ax = plotMeanDay(df_mean = df_mean, df_high = df_high, 
                          df_low = df_low, name = "", same_ax = same_ax)
        if save:
            fig.savefig(path2SaveFig+"allSeasons")
    
    # MEAN DAY PER SEASONS
    if (seasonal_sorting.count("SEASON")>0):
        df_seas=df2study.groupby(GroupMonthBySeas)
        # Calculate mean days by season
        for s in sorted(set(SeasonDict.values())):
            df_s_mean, df_s_high, df_s_low = meanDayCharac(df_seas.get_group(s), perc_values = perc_values)
            fig, ax = plotMeanDay(df_mean = df_s_mean, df_high = df_s_high, 
                        df_low = df_s_low, name = s, same_ax = same_ax)
            if save:
                fig.savefig(path2SaveFig+s)
                
    # MEAN DAY PER MONTHS
    if (seasonal_sorting.count("MONTH")>0):
        df_month=df2study.groupby(df2study.index.month)
        # Calculate mean days by season
        for m in sorted(set(df2study.index.month)):
            df_m_mean, df_m_high, df_m_low = meanDayCharac(df_month.get_group(m), perc_values = perc_values)
            fig, ax = plotMeanDay(df_mean = df_m_mean, df_high = df_m_high, 
                                  df_low = df_m_low, name = m, same_ax = same_ax)
            if save:
                fig.savefig(path2SaveFig+str(m))
    
    # Same as before for dimensionless days
    if (dimensionlessDay):
        # Calculate mean days by season AND using dimensionless days
        df_diml = {}
        df_diml["day"], df_diml["night"] = dimensionless(df2study, df_sun_events.copy(), night_adim = False)
        
        # MEAN FOR WHOLE YEARS USING DIMENSIONLESS DAYS
        if (seasonal_sorting.count("ALL")>0):
            # Calculate a dimensionless mean day and a dimensionless mean night and 
            # its variability around the mean and plot it
            perc_values = 0.5
            for p in df_diml.keys():
                df_mean = df_diml[p].groupby(level = 1).median()
                df_high = df_diml[p].groupby(level = 1).quantile(0.5+perc_values/2)
                df_low = df_diml[p].groupby(level = 1).quantile(0.5-perc_values/2)
                fig, ax = plotMeanDay(df_mean = df_mean, df_high = df_high, 
                                      df_low = df_low, name = "", same_ax = same_ax)
                if p == "night":
                    ax[-1].set_xlabel(u"Temps après coucher du soleil (min)")
                if save:
                    fig.savefig(path2SaveFig+"DimLess"+p+"_allSeasons")
        
        # MEAN FOR EACH SEASON USING DIMENSIONLESS DAYS
        if (seasonal_sorting.count("SEASON")>0):
            for p in df_diml.keys():
                df_diml_p_seas=df_diml[p].groupby(GroupMonthBySeas, level = 0)
                for s in sorted(set(SeasonDict.values())):
                        df_mean = df_diml_p_seas.get_group(s).groupby(level = 1).median()
                        df_high = df_diml_p_seas.get_group(s).groupby(level = 1).quantile(0.5+perc_values/2)
                        df_low = df_diml_p_seas.get_group(s).groupby(level = 1).quantile(0.5-perc_values/2)
                        fig, ax = plotMeanDay(df_mean = df_mean, df_high = df_high, 
                                              df_low = df_low, name = s, same_ax = same_ax)
                        if p == "night":
                            ax[-1].set_xlabel(u"Temps après coucher du soleil (min)")
                        if save:
                            fig.savefig(path2SaveFig+"DimLess"+p+"_"+s)
                        
        # MEAN FOR EACH MONTH USING DIMENSIONLESS DAYS
        if (seasonal_sorting.count("MONTH")>0):
            for p in df_diml.keys():
                df_diml_p_month=df_diml[p].groupby(pd.DatetimeIndex(df_diml[p].index.get_level_values(0)).month)
                for m in sorted(set(pd.DatetimeIndex(df_diml[p].index.get_level_values(0)).month)):
                        df_mean = df_diml_p_month.get_group(m).groupby(level = 1).median()
                        df_high = df_diml_p_month.get_group(m).groupby(level = 1).quantile(0.5+perc_values/2)
                        df_low = df_diml_p_month.get_group(m).groupby(level = 1).quantile(0.5-perc_values/2)
                        fig, ax = plotMeanDay(df_mean = df_mean, df_high = df_high, 
                                              df_low = df_low, name = m, same_ax = same_ax)
                        if p == "night":
                            ax[-1].set_xlabel(u"Temps après coucher du soleil (min)")
                        if save:
                            fig.savefig(path2SaveFig+"DimLess"+p+"_"+str(m))

def plotMeanDay(df_mean, df_high, df_low, name, same_ax = False):
    """Plot on a same Figure the mean day and its low and high equivalent.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df_mean : DataFrame
                Data of the mean day
            df_high : DataFrame
                Data of the high values day
            df_low : DataFrame
                Data of the low values day
            same_ax : Boolean, default False
                Whether or not the curves are plotted on a same axis

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			fig : Figure
                Figure where are plot the curves
            ax : Axis
                Axis in the figure where are plot the curves"""
    if (same_ax):
        fig, ax = plt.subplots(nrows = 1)
        
    else:
        fig, ax = plt.subplots(nrows = df_mean.columns.size, sharex = True)
    fig.suptitle(name)
    df_mean.plot(subplots = not same_ax, ax = ax, label = "median")
    if (df_mean.columns.size > 1)*(not same_ax):
        for axi in ax:
            axi.legend(loc = "lower left")
    else:
        ax.legend(loc = "lower left")
    
    if (same_ax):
        colors = [line.get_color() for line in ax.get_lines()]
    else:
        colors = None
    
    df_high.plot(subplots = not same_ax, ax = ax, linestyle = "--", color = colors, legend = False)
    df_low.plot(subplots = not same_ax, ax = ax, linestyle = "--", color = colors, legend = False)

    return fig, ax

def indicatorCalculation(df2study, df_W_int, df_sun_events, micromet_period_dic,
                         meteo_period_df, path2save = None, onlyMeteo=False):
    """Calculates average conditions for micro-meteorological data and meteorological
    data (returns only one of them if the other input is null):
        - Concerning micro-meteorological data, the averaging is performed
        for each day at one (or several) given periods of the day (and of the night).
        - Concerning meteorological data, the averaging is performed during a 
        given interval defined in hours before or after the begining of the
        micro-meteorological data period.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df2study : DataFrame
				The micro-meteorological data to process
            df_W_int : DataFrame
				The meteorological data to process
            df_sun_events : DataFrame
                The 'sunrise' and 'sunset' hours for each day of the 'df2study' dataset
            micromet_period_dic : dictionary of DataFrame
                For each key, a DataFrame contains all periods of the day or of
                the night that have to be study
			meteo_period_df : DataFrame
				The DataFrame contains for each micromet_period an interval of 
                values for the meteorological condition calculation. These
                periods are defined in hours before the begining of the micro-meteorological
                period.
            path2save : String, default None
                The URL where to save the indicator result for each season / period
                (if None not saved)
            onlyMeteo : boolean, default False
                Set to True if only the meteorological indicators
                (not the micro-meteorological) must be calculated

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			df_micromet_indic : dictionary
                The micro-meteorological indicators sorted by period and by month
            df_meteo_indic : dictionary
                The meteorological indicators sorted by period and by month"""        
    global SeasonDict
    
    # Calculation of the mean sunrise dates for each season (in minutes from 00:00)
    df_sun_sec = pd.DataFrame({se : pd.to_timedelta(df_sun_events[se].astype(str).values).total_seconds()
                    for se in df_sun_events.columns}, index = pd.DatetimeIndex(df_sun_events.index))
    df_sun_sec_s = df_sun_sec.groupby(GroupMonthBySeas)
    # Convert seconds to a time of the day round at dt = 15 minutes
    df_sun_s = {s : (pd.datetime(2019, 1, 1) + 
                pd.offsets.Second(int(df_sun_sec_s.get_group(s).median()["sunset"]))).round("900S")
                for s in df_sun_sec_s.groups.keys()}
    
    # Sort data by season for micro-meteo and meteo data
    if not onlyMeteo:
        df_seas = df2study.groupby(GroupMonthBySeas)
    df_W_int_seas = df_W_int.groupby(GroupMonthBySeas)
    
    # Calculate the median or mean for each period (and each day) of the micro-meteo and meteo data
    df_micromet_indic = {}
    df_meteo_indic = {}
    for p in meteo_period_df.index:
        print(p)
        filt_meteo = meteo_period_df.loc[p, :]
        df_micromet_indic[p] = {}
        df_meteo_indic[p] = {}
        for s in df_W_int_seas.groups.keys():
            print(s)
            # If the period of interest is during the day
            if p[0:2] == "PJ":
                filt_micro = [micromet_period_dic[s].loc[p, "period_start"], 
                              micromet_period_dic[s].loc[p, "period_end"]]
            # If the period of interest is during  night-time, time is in minutes 
            # from sunset time (rounded to the closest 15 minutes...)
            else:
                if ((micromet_period_dic[s].loc[p, "period_start"] is not np.nan)*
                    (micromet_period_dic[s].loc[p, "period_end"] is not np.nan)):
                    filt_micro = [str((df_sun_s[s] + pd.offsets.Minute(micromet_period_dic[s].loc[p, "period_start"])).round("900S").time())[0:5], 
                                  str((df_sun_s[s] + pd.offsets.Minute(micromet_period_dic[s].loc[p, "period_end"])).round("900S").time())[0:5]]
                else:
                    filt_micro = ["", ""]
            #Calculate the average value only if the period is not null or nan
            if (filt_micro[0] != "")*(filt_micro[1] != "")*(filt_micro != ""):
                if not onlyMeteo:
                    df_seas_s = df_seas.get_group(s)
                else:
                    df_seas_s = pd.DataFrame()
                df_W_int_seas_s = df_W_int_seas.get_group(s)
                df_micromet_indic[p][s], df_meteo_indic[p][s] = filterTimeAndAverage(df_seas_s, df_W_int_seas_s, 
                                                                                     filt_micro, filt_meteo, robust = False,
                                                                                     time_norm = False, onlyMeteo = onlyMeteo)

                if not df_micromet_indic[p][s].empty and path2save:
                    df_micromet_indic[p][s].to_csv(path2save+s+"_"+p+"_MicrometIndic.csv")
                if path2save:
                    df_meteo_indic[p][s].to_csv(path2save+s+"_"+p+"_MetIndic.csv")
            
                # Drop nan (for example for the first day if the period is already passed...)
                df_meteo_indic[p][s].dropna(inplace=True)
            
    return df_micromet_indic, df_meteo_indic
        

def plotMeteorologicalRelations(df_micromet_indic, df_meteo_indic, operations,
                                path2SaveFig, save = True, 
                                ratio2considerNotNan = 0.8, tick_size = 12):
    """Plot meteorological relations between micro meteorological and meteorological
    indicators.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df_micromet_indic : dictionary
                The micro-meteorological indicators sorted by period and by month
            df_meteo_indic : dictionary
                The meteorological indicators sorted by period and by month
            operations : DataFrame
                For each period of interest, contaings the list of the data
                needed to be study and the potential period used as reference
                for this period (substract the two periods)
            path2SaveFig : string
                Name of the URL where to save the resulting Figures if 'save' = True
            save : boolean, default True
                Whether or not the figures should be saved
            ratio2considerNotNan : float, default 0.8
                When averaging several temperature differences, ratio of temperature differences
                that should not be NaN
            tick_size : int, default 12
                Size of the tick axis

	Returns 
	_ _ _ _ _ _ _ _ _ _ 
df_sun_sec
			Return None"""
    for p in df_micromet_indic.keys():
        for s in df_micromet_indic[p].keys():
            print("\n\n\n" + p + " - " + s)
            fig, ax = plt.subplots(ncols = df_meteo_indic[p][s].columns.size, figsize = (20, 4), sharey = True)
            x = df_meteo_indic[p][s].copy()
            # Recover the station to average
            stations2averag = operations.loc[p, "list_average"]
            #Calculate the minimum number of stations that should not be nan in order to consider the time step
            thresh_nan = ceil(ratio2considerNotNan*len(stations2averag))
            y = df_micromet_indic[p][s][stations2averag].dropna(thresh = thresh_nan, axis = 1).mean(axis = 1)
            # Subtract the temperature difference of a previous time period in order to calculate whether the temperature
            # difference has increased or decreased
            if operations.loc[p, "subtract"] is not None:
                if s in df_micromet_indic[operations.loc[p, "subtract"]].keys():
                    y = y.subtract(df_micromet_indic[operations.loc[p, "subtract"]][s][operations.loc[p, "list_average"]].dropna(thresh = thresh_nan, axis = 1).mean(axis = 1))
                    y = y.reindex(x.index)
            print("x_col :" + x.columns)
            print("x_size :" + str(x.size))
            print("y_size :" + str(y.size))
            for i, var in enumerate(x.columns):
                ax[i].plot(x[var], y, "o", alpha = 0.25)
                fig, ax[i] = graph_layout(x_name = var, y_name = u"Temperature (K)",
                       fig = fig, ax = ax[i], x_tick_size = tick_size,
                       y_tick_size = tick_size, grid_on = False)
            #Display stuffs
            fig.suptitle("Period " + p + " - " + s)
            fig.subplots_adjust(left = 0.05, right = 0.99, wspace = 0.22, hspace = 0.30, bottom = 0.12, top = 0.90)
            if save:
                fig.savefig(path2SaveFig+p+"_"+s)


        
def identifyBestConfigurationTree(df_micromet_indic, df_meteo_indic, config_dic,
                                  path2SaveFig, save = True):
    """Plot meteorological relations between micro meteorological and meteorological
    indicators. Return the tree depth that maximizes the accuracy for each combination
    of season / period.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df_micromet_indic : dictionary
                The micro-meteorological indicators sorted by period and by month
            df_meteo_indic : dictionary
                The meteorological indicators sorted by period and by month
            config_dic : Dictionary
                Configuration dictionary for the study:
                    -> "ratio_calib": Ratio of the values to keep for the calibration
                    -> "n_eval": Number of evaluation (calibration / prediction processes)
                    -> "col_name": Name of the column containing the extremum / other class
                    -> "max_depth": Maximum depth of the tree to test
                    -> "df_conditions": Conditions for selecting the data for each period
                    -> "ratio2considerNotNan": when averaging several temperature
                    differences, ratio of temperature differences that should not be NaN
                    -> "tick_size": Size of the tick axis on the figure
                    -> "operations": For each period of interest, contaings 
                    the list of the data needed to be study and the potential 
                    period used as reference for this period (substract the two periods)
            path2SaveFig : string
                Name of the URL where to save the resulting Figures if 'save' = True
            save : boolean, default True
                Whether or not the figures should be saved

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			Return None"""
    # Some useful variables
    colors = pd.Series({"Summer": "red", "Spring": "green", "Winter": "blue", 
                        "Autumn" : "black"})
    ratio_calib = config_dic["ratio_calib"]
    ratio_calib = config_dic["ratio_calib"]
    n_eval = config_dic["n_eval"]
    col_name = config_dic["col_name"]
    max_depth = config_dic["max_depth"]
    df_conditions = config_dic["df_conditions"]
    ratio2considerNotNan = config_dic["ratio2considerNotNan"]
    tick_size = config_dic["tick_size"]
    
    fig, ax = plt.subplots(ncols = len(df_micromet_indic.keys()), figsize = (20, 4), sharey = True)
    fig.subplots_adjust(left = 0.05, right = 0.99, wspace = 0.22, hspace = 0.30, bottom = 0.12, top = 0.90)
    result = pd.DataFrame(columns = ["period", "season", "depth", "accuracy"])
    for axi, p in enumerate(df_micromet_indic.keys()):
        for si, s in enumerate(df_micromet_indic[p].keys()):
            print("\n\n\n" + p + " - " + s)
            
            meteo_var = df_conditions.loc[p, "meteorological_variables"]
            
            # Create the x and y objects that will be used in the model
            df_all = createXY(df_micromet_indic, df_meteo_indic, df_conditions, p,
                              s, col_name, ratio2considerNotNan)
            
            # Test the regression tree for several tree max_depth values
            accuracy_indexes = pd.MultiIndex.from_product([max_depth,range(0, n_eval)], names=['depth', 'iter'])
            accuracy = pd.Series(index = accuracy_indexes)
            for md in max_depth:
                # Perform 20
                for i in range(0, n_eval):
                    # Randomly select 'ratio_calib' of the data for calibration
                    df_calib = select_from_data(df_all, ratio_data = ratio_calib,
                                                distrib_col_name = col_name,
                                                final_distrib = "REPRESENTATIVE")
                    # Select the other part of the data for performance evaluation (verification)
                    df_verif = df_all.reindex(df_all.index.difference(df_calib.index))
                    
                    # Construct the decision tree
                    clf = tree.DecisionTreeClassifier(max_depth = md)
                    resulting_tree = clf.fit(df_calib[meteo_var], df_calib[col_name])
                    
                    # Predict the value from the tree using the verification dataset
                    predictions = resulting_tree.predict(df_verif[meteo_var])
                    
                    # Calculate the number of True estimations for this combination of max depth and iter
                    accuracybuff = (df_verif[col_name] == predictions).value_counts(normalize = True)
                    if accuracybuff.keys().contains(True):
                        accuracy.loc[md, i] = accuracybuff[True]
                    else:
                        accuracy.loc[md, i] = 0.0
            
            ylow = accuracy.median(level = 0) - [accuracy.loc[[(i0, i1) for i1 in range(0,20)]].quantile(0.25) 
                    for i0 in max_depth]
            yhigh = [accuracy.loc[[(i0, i1) for i1 in range(0,20)]].quantile(0.75) 
                     for i0 in max_depth] - accuracy.median(level = 0)
            label_accuracy = s + "(" + str(df_all[meteo_var].index.size) + " jours)"
            ax[axi].errorbar(x = pd.Series(max_depth) + float(si)/20, y = accuracy.median(level = 0),
                              yerr = [ylow, yhigh], marker = "o", label = label_accuracy, color = colors[s], fmt='o')
            fig, ax[axi] = graph_layout(x_name = "tree depth", y_name = u"Accuracy (%)",
                                       ax_title = "Period "+p, fig = fig, ax = ax[axi], x_tick_size = tick_size,
                                       y_tick_size = tick_size, grid_on = False,
                                       y_origin = None)
            ax[axi].legend()
    
            #Recover the tree depth that gives the most accurate results
            best_depth = accuracy.median(level = 0).idxmax()
            
            # Store this information in the result dataframe
            result = result.append(pd.Series({"period":p, "Season":s, "depth": best_depth,
                                              "accuracy": accuracy.median(level = 0).max()}),
                                    ignore_index = True)
            
            # Create the optimum decision tree and save it
            createDecisionTree(x = df_all[meteo_var], y = df_all[col_name], 
                               max_depth = best_depth, path2SaveFig = path2SaveFig,
                               filename = "Tree_"+s+p)
            
    if save:
        fig.savefig(path2SaveFig+"Accuracy.png")
        result.to_csv(path2SaveFig+"IdealTreeDepth.csv")

    return result


def createXY(df_micromet_indic, df_meteo_indic, df_conditions, period,
             season, col_name, ratio2considerNotNan = 0):
    """ Create the x (independent variables) and y (dependent variable) dataset.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

        		df_micromet_indic : pd.Series
                The dependent variable data
            df_meteo_indic : pd.DataFrame
                The independent variable data
            df_conditions : pd.DataFrame
                Informations about operations to apply on y values (for each period as index). 
                Must contain as columns:
                    -> "list_average": List of stations to average as y
                    -> "subtract": Period to use as reference if an 'evolution of y' is calculated (None if not)
                    -> "extremum_type": "MIN" or "MAX"
                    -> "quantile_to_keep": Ratio of extremum values to keep
            period : String
                Name of the period of the day
            season : string
                Name of the season
            col_name : String
                Name to give to the y value (dependent variable)
            ratio2considerNotNan : float, default 0
                Ratio of stations needed to consider a time step as "usable" (all if 0)

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			Return the x and y variables"""      
    # List of meteorological variables to keep for the model
    meteo_var = df_conditions.loc[period, "meteorological_variables"]
    
    x = df_meteo_indic[period][season][meteo_var].copy()
    
    # Ratio of extremum values to keep
    ratio_of_val_to_keep = df_conditions.loc[period, "quantile_to_keep"]
    
    # Recover the station to average
    stations2averag = df_conditions.loc[period, "list_average"]

    #Calculate the minimum number of stations that should not be nan in order to consider the time step
    thresh_nan = ceil(ratio2considerNotNan*len(stations2averag))
    
    y = df_micromet_indic[period][season][stations2averag].dropna(thresh = thresh_nan, axis = 1).mean(axis = 1)
    
    # Subtract the temperature difference of a previous time period in order to calculate whether the temperature
    # difference has increased or decreased
    if df_conditions.loc[period, "subtract"] is not None:
        if season in df_micromet_indic[df_conditions.loc[period, "subtract"]].keys():
            y = y.subtract(df_micromet_indic[df_conditions.loc[period, "subtract"]][season][stations2averag].dropna(thresh = thresh_nan, axis = 1).mean(axis = 1))
    
    df_all = x.join(pd.Series(y, name = "y"), how = "inner").dropna(how = "any")      
    
    # Set a different class to the samples being extremum and other values
    if df_conditions.loc[period, "extremum_type"] == "MAX":
        index_extremum = df_all[df_all.y>=df_all.y.quantile(1-ratio_of_val_to_keep)].index
        index_other = df_all[df_all.y<df_all.y.quantile(1-ratio_of_val_to_keep)].index
        df_all.loc[index_extremum, col_name] = ["extremum" for ind in index_extremum]
        df_all.loc[index_other, col_name] = ["other" for ind in index_other]
    elif df_conditions.loc[period, "extremum_type"] == "MIN":
        index_extremum = df_all[df_all.y<=df_all.y.quantile(ratio_of_val_to_keep)].index
        index_other = df_all[df_all.y>df_all.y.quantile(ratio_of_val_to_keep)].index
        df_all.loc[index_extremum, col_name] = ["extremum" for ind in index_extremum]
        df_all.loc[index_other, col_name] = ["other" for ind in index_other]
    df_all.drop(["y"], axis = 1, inplace = True)
    
    return df_all

def createDecisionTree(x, y, max_depth = None, path2SaveFig = None, filename = None):
    """ Create a decision tree according to a x / y dataset and a tree depth.
    Can save the tree as a PNG.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			y : pd.Series
                The dependent variable data
            x : pd.DataFrame
                The independent variable data
            max_depth : integer, default None
                Maximum depth of the tree to build (if None, no maximum)
            path2SaveFig : string, default None
                Name of the URL where to save the tree as image (if None, no figure plotted)
            filename : string, default None
                Name of the file to save the tree as image (if None, no figure plotted)

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			Return a decision tree classifier"""    
    x_cols = x.columns
    
    #Merge x and y and remove nan
    df_all = x.join(pd.Series(y, name = "y"), how = "inner").dropna(how = "any")      
    
    # Create the tree with all data for this season / period of day
    clf = tree.DecisionTreeClassifier(max_depth = max_depth)
    final_tree = clf.fit(df_all[x_cols], df_all.y)
    
    # Save the model into a joblib file if a path is indicated
    if path2SaveFig and filename:
        dump(final_tree, path2SaveFig+filename+'.joblib') 
    
        # Export the joblist tree as an image (.dot)
        dot_data = tree.export_graphviz(final_tree, out_file=None, 
                                        feature_names=x.columns,  
                                        class_names=sorted(set(y)),  
                                        filled=True, rounded=True,  
                                        special_characters=True)
    
        graphviz.Source(dot_data).save(filename = filename+".dot",
                                       directory = path2SaveFig)
        (ImageTree,) = pydot.graph_from_dot_file(path2SaveFig+filename+".dot")
        ImageTree.write_png(path2SaveFig+filename+".png")
        
        os.remove(path2SaveFig+filename+".dot")
        os.remove(path2SaveFig+filename+".joblib")
        
    return final_tree

def createYearsAndMonthsDic(df):
    """ Create a dictionary of years and the corresponding months that are present
    in the micro-meteorological data ("df").
    
        Pameters
		_ _ _ _ _ _ _ _ _ _ 
							
			self : pandas.DataFrame
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
                             h, m) 
                for h in result_mean.index.levels[0]\
                for m in result_mean.index.levels[1]]
    result_mean.index = new_index
    result_high.index = new_index
    result_low.index = new_index
    
    return result_mean, result_high, result_low

def filterTimeAndAverage(df, df_W, filt_micro, filt_W, robust = False,
                         time_norm = False, onlyMeteo = False):
    """ Acts differently on the two input types:
            - for micro-meteorological data, filters a certain range of
            the day time (e.g.: ["8:15", "10:15"] or ["23:00", "01:00"])
            and average it to have a single daily value.
            - for each meteorological variable (wind speed, nebulosity, etc.), 
            filters a specific range of hours before the micro-meteorological
            filtering (e.g.: {"Tair": [-2, 2], "RHair": [-3, 2], 
                              "WindSpeed": [0, 2], 
                              "Nebulosity": [-10, -2],...})
    
    WARNING : does not work if the meteorological filtering period ends more than 24 hours
    before the begining of the micrometeorological period...

        Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
			df : pandas.DataFrame
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
            onlyMeteo : boolean, default False
                Set to True if only the meteorological indicators
                (not the micro-meteorological) must be calculated
            
        Returns
		_ _ _ _ _ _ _ _ _ _ 
							
        		One or two dataframes : the first being the mean micro-meteorological values
                during the given interval for each day, the second the mean
                meteorological condition for each meteorological interval (and for
                each day)"""
    if not onlyMeteo:
        # Filters the micro-meterological data
        df_micro_filt = df.between_time(pd.Timestamp(filt_micro[0]).time(),\
                                        pd.Timestamp(filt_micro[1]).time())
                    
        # Average the interval for each day (starting the average at the beginning of
        # the interval)
        if robust == True:
            df_micro_result = df_micro_filt.resample("24H").median()
        else:
            df_micro_result = df_micro_filt.resample("24H").mean()
    
        # Creates the dataframe that will recover results
        df_W_result = pd.DataFrame(index = df_micro_result.index)
        
    else:
        # Creates the dataframe that will recover results
        df_W_result = pd.DataFrame(index = sorted(set(df_W.index.date)))
        
    # For each meteorological variable
    for v in filt_W.keys():
        # Use moving average to average (by median or mean) values using a "NB_hours" 
        # interval defined as the duration in the "filt_W" period
        if robust == True:
            # Note that rolling for offset automatically stored the rolling average to the right (end of the interval)
            df_W_avg = df_W[v].rolling(pd.offsets.Hour(filt_W[v][1]-filt_W[v][0])).median()
        else:
            df_W_avg = df_W[v].rolling(pd.offsets.Hour(filt_W[v][1]-filt_W[v][0])).mean()
            
        # The moving average, while it averages over time, keeps one value for each time step
        # Here we conserve only the value corresponding to the last time step of the "filt_W" period
        t_end_range = (pd.Timestamp(filt_micro[0]) + pd.offsets.Hour(filt_W[v][1])).time()
        df_W_filt = df_W_avg.at_time(t_end_range)
        
        # In order to attribute meteorological values to the right studied day
        # it is necessary to offset df_W values under certain conditions
        micro_start_hour = int(filt_micro[0].split(":")[0])
        # the end of df_W average is in a previous day
        if micro_start_hour < -filt_W[v][1]:
            df_W_filt.index = pd.DatetimeIndex(df_W_filt.index.date) + pd.offsets.Day(1)
        # the end of df_W average is in the next day
        elif 24-micro_start_hour < filt_W[v][1]:
            df_W_filt.index = pd.DatetimeIndex(df_W_filt.index.date) + pd.offsets.Day(-1)
        else:
            df_W_filt.index = df_W_filt.index.date
        # Add the result for this specific variable in a DataFrame containing 
        # all variables (using an intersection of the indexes)
        df_W_result = df_W_result.join(df_W_filt, how = "inner")
    
    if not onlyMeteo:
        # Reindex the microclimat data in order to have the same indexes as the meteorological data
        df_micro_result = df_micro_result.reindex(df_W_result.index)

    else:
        df_micro_result = pd.DataFrame()
    
    return df_micro_result, df_W_result

def getSunEvents(df, location):
    """ Get the sunrise and sunset time for all days corresponding to a given
    dataset (plus the previous day and the one after).
    		
        Parameters
		_ _ _ _ _ _ _ _ _ _ 

				df : pandas.DataFrame
					Object where are contained the climatic data.
                location : geopy.geocoders.Nominatim().geocode(city) object
                    Object containing all geographical informations of a place
                    on the globe (such as longitude, latitude)

		Returns
		_ _ _ _ _ _ _ _ _ _ 

        			Return a pandas DataFrame containing as index a list of day and
                two columns: the first consists of the sunrise time, 
                the second of the sunset ones."""
    # Load a timescale object and also an ephemeris file that provides positions 
    # from the planets:
    tim_sc = api.load.timescale()
    planets = api.load('de421.bsp')

    # Recover the begining and end of the climatic data and get the previous and the next days
    start_time = df.sort_index().index[0] - pd.offsets.Day(1)
    end_time = df.sort_index().index[-1] + pd.offsets.Day(1)
    
    # Convert these dates into skyfield corresponding objects
    sta_utc = tim_sc.utc(start_time.year, start_time.month, start_time.day)
    end_utc = tim_sc.utc(end_time.year, end_time.month, end_time.day + 1)
    
    # Create a Topos object describing your geographic location
    top_obj = api.Topos(location.latitude, location.longitude)
    
    # Recover a list of sunrise and sunset and its corresponding boolean list
    # (True for sunrise only)
    times, boolean = almanac.find_discrete(sta_utc, end_utc, \
                                           almanac.sunrise_sunset(planets, top_obj))
    
    # Convert the lists into a DataFrame
    df_raw = pd.DataFrame({"dates": times.utc_iso(), "boolean": boolean})
    # Convert the dates to datetime objects
    df_raw["dates"] = pd.to_datetime(df_raw['dates'])
    
    df_result = pd.DataFrame({"sunrise": pd.DatetimeIndex(df_raw[df_raw.boolean == True].dates).time,\
                              "sunset": pd.DatetimeIndex(df_raw[df_raw.boolean == False].dates).time},
                            index = sorted(set(pd.DatetimeIndex(df_raw.dates).date)))
    
    return df_result

def getSunAngle(df_index, location):
    """ Get the position of the sun for a given set of dates
    from an horizontal reference (positive if the sun can be seen by the observer).

		Parameters
		_ _ _ _ _ _ _ _ _ _ 

				df_index : pandas.date_range
					Object where are contained the times where we want to calculate
                the sun position.
            location : geopy.geocoders.Nominatim().geocode(city) object
                Object containing all geographical informations of a place
                on the globe (such as longitude, latitude)                
            
		Returns
		_ _ _ _ _ _ _ _ _ _ 

				Return a pandas.Series containing a sun position (in degree)."""
    # Load a timescale object and also an ephemeris file that provides positions 
    # from the planets:
    tim_sc = api.load.timescale()
    planets = api.load('de421.bsp')
    
    # Create a Topos object describing your geographic location
    top_obj = planets["earth"] + api.Topos(location.latitude, location.longitude)  

    # Calculates the sun position for each index
    result = pd.Series(top_obj.at(tim_sc.from_datetimes(df_index)).\
                            observe(planets["SUN"]).apparent().altaz()[0].radians, \
                        index = df_index)
    
    return result

def dimensionless(df, df_th_sun, night_adim = False, dt = 15):
    """Return a PanelDimensionlessDay object with a normalised daytime index. 
    The time index is based on the day time (a day : [0 - 1]) and the night time
    index can either be in minutes after the sunset, either also adimensionned (a night : [0 - 1])
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			self : ModifiedData object
				Object where are contained the measurements
			df_th_sun : pd.DataFrame
				Theoretical solar radiation (W / m2) - define the border between day and night-time;
			night_adim : boolean
				Whether or not the night time is adimensionned or keep in minutes after sunset
            dt : integer, default 15
                Time step of the initial signal (in minutes)

	Returns
	_ _ _ _ _ _ _ _ _ _ 

			Return PanelDimensionlessDay object with a new time index."""
    df_night = df.copy()
    df_night.index = df_night.index - pd.offsets.Hour(12)			#To enable the entire night to be at a same date (in a same day)
    df_th_sun.loc[:,"sunset"] = pd.to_datetime(df_th_sun.index.astype(str) + ' ' + df_th_sun["sunset"].astype(str))
    df_th_sun.loc[:,"sunrise"] = pd.to_datetime(df_th_sun.index.astype(str) + ' ' + df_th_sun["sunrise"].astype(str))
    
    result_day = {}
    result_night = {}
    index_fin_day = pd.Index(np.linspace(0, 1, 100, endpoint = False))
    #For each day
    for d in sorted(set(df.index.date)):
        buff_day = df[str(d)].copy()
        print(d)
        # DO THE DAY-TIME TRANSFORMATION
        # Process the day only if there are values
        if buff_day.dropna(how = "all").size != 0:
            buff_day_day = buff_day[(buff_day.index >= df_th_sun.loc[d, "sunrise"])*
                                    (buff_day.index < df_th_sun.loc[d, "sunset"])]
            # Process the day only if there are values
            if buff_day_day.dropna(how = "all").size != 0:                
                buff_day_day.index = pd.Index(np.linspace(0, 1, len(buff_day_day.index), endpoint = False))
    
                buff_day_day = buff_day_day.reindex(buff_day_day.index.union(index_fin_day))
                buff_day_day = buff_day_day.astype(float)
                buff_day_day = interpolate_val(buff_day_day).reindex(index_fin_day)
            else:
                buff_day_day = pd.DataFrame(columns = buff_day.columns)

            result_day[d] = buff_day_day.copy()

    # DO THE NIGHT-TIME TRANSFORMATION
    for d in sorted(set(df_night.index.date)):
        buff_day = df_night[str(d)].copy()
        # Process the day only if there are values
        if buff_day.dropna(how = "all").size != 0:
            buff_day_night = buff_day[(buff_day.index < df_th_sun.loc[d, "sunrise"] 
                        + pd.offsets.Day(1) - pd.offsets.Hour(12))*
                        (buff_day.index >= df_th_sun.loc[d, "sunset"]
                        - pd.offsets.Hour(12))]
            
            # IF THE NIGHT-TIME SHOULD BE ADIMENSIONNED
            if night_adim:
                index_fin_night = pd.Index(np.linspace(0, 1, 100, endpoint = False))
                # Process the day only if there are values
                if buff_day_night.dropna(how = "all").size != 0:
                    buff_day_night.index = pd.Index(np.linspace(0, 1, len(buff_day_night.index), endpoint = False))
                    
                    buff_day_night = buff_day_night.reindex(buff_day_night.index.union(index_fin_night))
                    buff_day_night = buff_day_night.astype(float)
                    buff_day_night = interpolate_val(buff_day_night).reindex(index_fin_night)
                else:
                    buff_day_night = pd.DataFrame(columns = buff_day.columns)
            
                result_night[d] = buff_day_night.copy()

            # WE SET THE SUNSET AT 0 AND THEN INDEX THE NIGHT IN MINUTES
            else:
                # Process the night only if there are values
                if buff_day_night.dropna(how = "all").size != 0:
                    buff_day_night.index = buff_day_night.index - pd.offsets.Hour(buff_day_night.index[0].hour) - pd.offsets.Minute(buff_day_night.index[0].minute)
                    buff_day_night.index = buff_day_night.index.hour*60 + buff_day_night.index.minute + buff_day_night.index.second/60
                    buff_day_night = buff_day_night.astype(float)
                else:
                    buff_day_night = pd.DataFrame(columns = buff_day.columns)
    							
                result_night[d] = buff_day_night.append(buff_day_night)
	
    #Panels are deprecated, instead convert the dictionaries into pandas multiindex dataframes
    result_df_day = pd.concat(result_day.values(), keys = result_day.keys())
    result_df_night = pd.concat(result_night.values(), keys = result_night.keys())
    #Verify that all columns are set to numeric
    for c in result_df_day.columns:
        result_df_day[c] = result_df_day[c].astype(float)
        result_df_night[c] = result_df_night[c].astype(float)
        
    return result_df_day, result_df_night

def graph_layout(x_name, y_name, fig = None, ax = None, y_range = [], x_range = [], x_tick_size = 25, y_tick_size = 25, x_origin = None, y_origin = 0., color = "black", \
linewidth = 2, ax_title = "", title_position = "center", grid_on = True):
	"""Define several parameters for a given figure and axe.
	
		Parameters
	_ _ _ _ _ _ _ _ _ _ 
	
			x_name : str
				Horizontal axis name (and unit)
			y_name : str
				Vertical axis name (and unit)
			fig : plt.Figure object, default None
				Figure to be modified
			ax : plt.Axes object, default None
				Axes to be modified
			y_range : list of floats, default []
				The vertical axis values can be defined (values which will appear on the vertical-axis)
			x_range : list of floats, default []
				The horizontal axis values can be defined (values which will appear on the horizontal-axis)
			x_tick_size : int, default 25
				Font size for the x-axis values and x label
			y_tick_size : int, default 25
				Font size for the y-axis values and y label
			x_origin : float, default None.
				Value of the horizontal axis for which will be drawn a bold vertical line (horizontal axis reference)
			y_origin : float, default 0.
				Value of the vertical axis for which will be drawn a bold horizontal line (vertical axis reference)
			color : plt.color object, default "black"
				Color to use for vertical and horizontal origins
			linewidth : float, default 2
				Thickness of the grid to show
			ax_title : str, default ""
				Title for the axis
			title_position : {"center", "left", "right"}, default = "center"
				Where will be written the ax title
			grid_on : boolean, default True
				Whether or not the grid is displayed
				
				
		Returns
	_ _ _ _ _ _ _ _ _ _ 
	
			Modified Figure and axes input objects """
	
	if fig == None:
		fig = plt.figure(plt.get_fignums()[-1])
	if ax == None:
		ax = fig.get_axes()[0]
	
	if len(y_range) != 0:
		ax.set_yticks(y_range)
	if len(x_range) != 0:
		ax.set_xticks(x_range)
		#Set the color of the y-axis values
	for i in ax.get_yticklabels():
		i.set_color(color)
	ax.tick_params(axis = "y", labelsize = y_tick_size, color = color)
	ax.tick_params(axis = "x", labelsize = x_tick_size)
	ax.set_xlabel(x_name, fontsize = x_tick_size)
	ax.set_ylabel(y_name, fontsize = y_tick_size, color = color)
	if y_origin is not None:
		ax.axhline(y_origin, linewidth = 2, color = color)
	if x_origin is not None:
		ax.axvline(x_origin, linewidth = 3, color = color)
	if grid_on is True:
		ax.grid("on", linewidth = linewidth, color = color)		#Set a grid with dashed line
	fig.patch.set_facecolor('white')		#Set a white background for the figure instead of a grey one
	ax.set_title(ax_title, loc = title_position, fontsize = 15)
	
	return fig, ax
    
def interpolate_val(df):
	last_valid = df.dropna(how = "all").index[-1]
	result = df.interpolate(method = "index")
	result[result.index > last_valid] = pd.DataFrame(columns = df.columns, index = result[result.index > last_valid].index)
	
	return result

def RH_to_PressureDeficit(RH, T):
    """ Calculate a 'deficit' of vapour pressure to evaluate the 
    evaporation potential (vapour pressure - saturating vapor pressure)
    		
			Parameters
			_ _ _ _ _ _ _ _ _ _ 
							
					T : pd.Series
						Series containing temperature data (in °C)
					RH : pd.Series
						Series containing relative humidity data (in %)
						
			Returns
			_ _ _ _ _ _ _ _ _ _ 
			
					A pd.Series containing the Vapour pressure (hPa)"""
    return RH_to_Vp(T, RH).mul(1-RH/100)

def RH_to_Vp(Temp, RHum):
	"""Calculate the Vapour pressure (hPa) from relative humidity (%) and temperature (°C) data (W. Wagner and A. Pruß:" The IAPWS Formulation 1995 for the Thermodynamic Properties
	of Ordinary Water Substance for General and Scientific Use ", Journal of Physical and Chemical Reference Data, June 2002 ,Volume 31, Issue 2, pp. 387535
		
			Parameters
			_ _ _ _ _ _ _ _ _ _ 
							
					Temp : pd.Series
						Series containing temperature data (in °C)
					RHum : list
						Series containing relative humidity data (in %)
						
			Returns
			_ _ _ _ _ _ _ _ _ _ 
			
					A pd.Series containing the Vapour pressure (hPa)"""
		
		#Parameter definition for calculation
	TK = Temp + 273.15
	Tc = 647.096			#Critical temperature
	Pc = 220640			#Critical pressure
		#Coefficients
	C1 = -7.85951783
	C2 = 1.84408259
	C3 = -11.7866497
	C4 = 22.6807411
	C5 = -15.9618719
	C6 = 1.80122502
	
		#Conversion from relative humidity to vapour pressure
	V = 1 - TK / Tc
	Pws = Pc * np.exp(Tc / TK * (C1 * V + C2 * V**1.5 + C3 * V**3 + C4 * V**3.5 +C5 * V**4 + C6 * V**7.5))
	result = RHum * Pws / 100
	
	return result



def select_from_data(df, nb_data = np.nan, ratio_data = 0.7, nb_data_class = np.nan, distrib_col_name = \
                     "I_TYPO", classif_col_name = None, final_distrib = "REPRESENTATIVE"):
    """ Select a random sample of 'nb_data' (number of building) respecting the 
    'final_distribution' of 'distrib_col_name' (building typology) for each 
    'classif_col_name' (city).
    
        Parameters
    _ _ _ _ _ _ _ _ _ _ 
    
            df : pd.DataFrame
                Object containing the initial data from which the selection should be performed
            nb_data : int, default "NaN"
                Number of sample to keep (in case 'final_distrib' = "REPRESENTATIVE" and 'ratio_data' = np.nan)
            ratio_data : int, default 0.7
                Ratio of sample number to keep (in case 'final_distrib' = "REPRESENTATIVE" and 'nb_data' = np.nan)
            nb_data_class : int, default "NaN"
                Number of building per class (in case 'final_distrib' = "EQUALLY")
            distrib_col_name : string, default "I_TYPO"
                Name of the column used for the distribution
            classif_col_name : string, default None
                Name of the column for which nb_data should be conserved
            final_distrib : {"REPRESENTATIVE", "EQUALLY", "RANDOM"}, default "REPRESENTATIVE"
                Type of distribution of the distrib_col_name variable between the input and the output data
                    -> "REPRESENTATIVE" : The percentage of data belonging to a specific class ('distrib_col_name' column)
                    should be equal in the input and the output data
                    -> "EQUALLY" : The number of data should be similar between all classes in the output data
                    -> "RANDOM" : The distribution inside each class is not taken into account for the choice
        Returns
    _ _ _ _ _ _ _ _ _ _ 
    
            The selected data (pd.DataFrame object)"""
    result = pd.DataFrame(columns = df.columns)
    
    # Whether of not the random sampling should be done identically for 
    # each class of the 'classif_col_name' column
    if classif_col_name is None:
        cities = [None]
    else:
        cities = sorted(set(df[classif_col_name]))
    if final_distrib is not "RANDOM":
        classes = sorted(set(df[distrib_col_name]))   
    
    if final_distrib is "REPRESENTATIVE":
        # The data selection is performed differently if 'nb_data' is fullfilled
        if nb_data is not np.nan:
            data_selection_crit = nb_data
            normalize = True
        else:
            data_selection_crit = ratio_data
            normalize = False 
        for c in cities:
            if c is None:
                buff = df.copy()
            else:
                buff = df[df[classif_col_name] == c]
            for cl in sorted(set(buff[distrib_col_name])):
                buff_class = buff[buff[distrib_col_name] == cl].copy()
                nb2keep = int(buff[distrib_col_name].value_counts(normalize = normalize)[cl] * data_selection_crit)
                data2add = buff_class.loc[random.sample(buff_class.index, \
                                                        nb2keep),:].copy()
                result = result.append(data2add)

    elif final_distrib is "EQUALLY":
        for c in cities:
            if c is None:
                buff = df.copy()
            else:
                buff = df[df[classif_col_name] == c]
            for cl in sorted(set(buff[distrib_col_name])):
                buff_class = buff[buff[distrib_col_name] == cl].copy()
                data2add = buff_class.loc[random.sample(buff_class.index, nb_data_class),:].copy()
                result = result.append(data2add)
    
    elif final_distrib is "RANDOM":
        for c in cities:
            if c is None:
                buff = df.copy()
            else:
                buff = df[df[classif_col_name] == c]
            data2add = buff.loc[random.sample(buff.index, nb_data),:].copy()
            result = result.append(data2add) 
    
    return result


def diffuseRadiationFromReindl1990(I,phi,Ta, location):    
    # Intersect the indexes from the three series in order to keep
    # only indexes which will result in having a value
    index = I.index.intersection(phi.index).intersection(Ta.index)
    
    # Gather variables in a DataFrame to simplify manipulations
    df = pd.concat([I.rename("I"), phi.rename("phi"), Ta.rename("Ta")],
                   axis = 1).reindex(index)
    
    # Calculate the solar altitude angle for each of the index
    df["alpha"] = getSunAngle(index, location)
    
    # Calculate the extraterrestrial global radiation on a horizontal
    # surface for the given indexes and location on earth
    Iextra = getExtraTerrestrialSolarRadiation(df["alpha"])
    
    # Calculate the clearness index
    df["kt"] = df.I.divide(Iextra)
    
    # If data sampled more frequently than 1h,
    # use a rolling average to have hourly values
    # (especially hourly radiations) for each time step
    nb_sample_per_hour = int(pd.to_timedelta("1H")/\
                             pd.infer_freq(I.index))
    if(nb_sample_per_hour>1):
        df = df.rolling(pd.offsets.Hour(1),
                      min_periods = nb_sample_per_hour).mean()
    
    # Filter the indexes according to the kt threshold 
    # in order to apply relations from the piece-wise correlation
    df_a = df[(df.kt>=0)&(df.kt<=0.3)]
    df_b = df[(df.kt>0.3)&(df.kt<0.78)]
    df_c = df[df.kt>=0.78]
    
    # Apply the relations for each correlation
    Id_a = df_a.I.multiply(1.000-0.232*df_a.kt + 0.0239*np.sin(df_a.alpha)-\
                           0.000682*df_a.Ta + 0.0195*df_a.phi)
    Id_b = df_b.I.multiply(1.329-1.716*df_b.kt + 0.267*np.sin(df_b.alpha)-\
                           0.00357*df_b.Ta + 0.106*df_b.phi)/10
    Id_c = df_c.I.multiply(0.426*df_c.kt + 0.256*np.sin(df_c.alpha)-\
                           0.00349*df_c.Ta + 0.0734*df_c.phi)/10
    
    # Remove potential bias according to Reindl constraints
    Id_a[Id_a>I.reindex(Id_a.index)] = I.reindex(Id_a.index)[Id_a>I.reindex(Id_a.index)]
    Id_b[Id_b<0.1*I.reindex(Id_b.index)] = 0.1*I.reindex(Id_b.index)[Id_b<0.1*I.reindex(Id_b.index)]
    Id_b[Id_b>0.97*I.reindex(Id_b.index)] = 0.97*I.reindex(Id_b.index)[Id_b>0.97*I.reindex(Id_b.index)]
    Id_c[Id_c<0.1*I.reindex(Id_c.index)] = 0.1*I.reindex(Id_c.index)[Id_c<0.1*I.reindex(Id_c.index)]
    Id_c[Id_c>0.97*I.reindex(Id_c.index)] = 0.97*I.reindex(Id_c.index)[Id_c>0.97*I.reindex(Id_c.index)]
       
    """# Merge the series pieces into one 
    Id = Id_a.append(Id_b).append(Id_c)
    
    return Id"""
    return Id_a, Id_b, Id_c, df.kt
    
def getExtraTerrestrialSolarRadiation(sun_angle):
    # Solar constant (W/m²)
    solarConstant = 1366.1
    
    return  solarConstant * \
            (1+0.033*np.cos(2*np.pi*sun_angle.index.dayofyear/365)) *\
            np.sin(sun_angle)