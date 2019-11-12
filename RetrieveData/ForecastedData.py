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
    cloud covering, air relative humidity and atmospheric pressure.

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
    col_names = {"toSet": ["Date", "ImageWeather", "Air temperature", "ImageWind dir",
                           "Wind speed", "Gust wind speed", "Precipitations", 
                           "Cloud cover", "Air relative humidity", 
                           "Atmospheric pressure"],
                 "toKeep": ["Air temperature", "Wind speed", 
                            "Precipitations", "Cloud cover", 
                            "Air relative humidity", "Atmospheric pressure"]}
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
    for i in range(3, 12):
        print i
        # Replace the column names and keep only those useful
        df_list[i].columns = col_names["toSet"]
        df_list[i] = df_list[i][col_names["toKeep"]]
        # Create the datetime column to replace the jour by a date and the hour
        df_list[i].index = [pd.datetime.today().date() + pd.offsets.Day(i-3) +\
                                       pd.offsets.Hour(h) for h in df_list[i].index]
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
    
    return pd.DataFrame(df_output, dtype = float)