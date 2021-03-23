#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:54:28 2019

@author: Jérémy Bernard
"""

import requests
import pandas as pd

def downloadMeteorama(city = 'Grièges',
                      output_directory = "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP/",
                      urlBeforeCity="https://www.meteorama.fr/météo-",
                      urlAfterCity=".html?v=heure-par-heure"):
    """ Download hourly weather forecasted for the next 10 days for a given
    city (based on Meteorama forecasting: https://www.meteorama.fr). Available
    meteorological variables are: air temperature, wind speed, precipitations,
    cloud covering, air relative humidity and atmospheric pressure. Convert all
    of them in current units used in meteorology.

        	Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
			city : string, default 'Grièges'
				dictionary containing as keys the years and as values the corresponding
                month to download
			output_directory : string, default "/tmp/forecastedMeteorama/"
				string where is stored the base of the URL construction used
                to download the data
            urlBeforeCity : string, default "https://www.meteorama.fr/météo-"
                string being part of the URL prefix name to download the data
            urlAfterCity : string, default ".html?v=heure-par-heure"
                string being part of the URL suffix name to download the data

		Returns
		_ _ _ _ _ _ _ _ _ _ 
							
			A DataFrame containing air temperature, wind speed, precipitations,
            cloud covering, air relative humidity and atmospheric pressure for
            the next 10 days for the "city" supplied as input."""
    
    # Column names to set and to keep for the data recovered
    col_names = {"toSet": ["Date", "ImageWeather", "Tair", "ImageWind dir",
                           "WindSpeed", "WindDir", "Precipitations", 
                           "Nebulosity", "RHair", "Patmo"],
                 "toKeep": ["Tair", "WindSpeed", "Precipitations", 
                            "Nebulosity", "RHair", "Patmo"]}
    # String to remove from the data recovered on the internet
    char2remove = [u'\xc2\xba', " km/h", " mm", "%", "%", " hPa"]

    # Construct the URL appropriate for the given city
    url = urlBeforeCity + city.lower().replace(" ", "-") + urlAfterCity
    
    # Need to simulate a web browser in order to be able to download html file
    # from Meteorama web-site
    header = {
      "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
      "X-Requested-With": "XMLHttpRequest"
    }
    r = requests.get(url, headers=header)
    
    # Get all html tables from the Meteorama web-site and convert them into a
    # list of csv tables
    df_list = pd.read_html(r.text)

    # Create the final DataFrame from each DataFrame day
    df_output = pd.DataFrame(columns = col_names["toKeep"])
    for i in range(0, len(df_list)):
        # Replace the column names and keep only those useful
        df_list[i].columns = col_names["toSet"]
        df_list[i] = df_list[i][col_names["toKeep"]]
        # Set the data to unicode in case it is integer (it is the case when the nebulosity is 0 for the whole day...)
        df_list[i] = df_list[i].astype("unicode")
        
        # Create the datetime column to replace the day by a date and the hour
        df_list[i].index = [pd.datetime.today().date() + pd.offsets.Day(i) +\
                                       pd.offsets.Hour(23-df_list[i].index.max()+h) for h in df_list[i].index]
        
        df_list[i]["Nebulosity"]
        
        # Remove characters that are not numbers for each column
        for j, var in enumerate(char2remove):
            df_list[i][df_list[i].columns[j]] = df_list[i][df_list[i].columns[j]]\
                .map(lambda x: x.rstrip(char2remove[j]))
            # Replace ' by . for precipitations values...
            if var == " mm":
                df_list[i][df_list[i].columns[j]] = df_list[i][df_list[i].columns[j]]\
                .map(lambda x: x.replace(",", "."))
        
        #Add the current day to the global DataFrame
        df_output = df_output.append(df_list[i])
    
    #Convert all data to float
    df_output = pd.DataFrame(df_output, dtype = float)
    
    # Convert to international units (m/s and Pa)
    df_output["WindSpeed"] = df_output["WindSpeed"] / 3.6
    df_output["Patmo"] = df_output["Patmo"] * 100
    
    return pd.DataFrame(df_output, dtype = float)
    
