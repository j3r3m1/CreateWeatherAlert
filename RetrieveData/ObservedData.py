#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:54:28 2019

@author: Jérémy Bernard
"""

import wget
import zlib
import pandas as pd

def downloadMeteoFranceSynop(yearsAndMonths = {2018: range(1, 13), 2019: range(1, 10)},
                            output_directory = "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP/",
                            url="https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/",
                            baseName="synop.",
                            fileFormat=".csv.gz"):
    """ Download the data available on the Meteo-France web-site :
        https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32
        The data are recovered and saved in monthly files.

        	Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
			yearsAndMonths : dictionary, default [2018: [range(1, 13)], 2019: range(1, 10)]
				dictionary containing as keys the years and as values the corresponding
                month to download
			output_directory : string, default "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP"
				string where is stored the base of the URL construction used
                to download the data
            baseName : string, default "synop."
                string being part of the URL base name to download the data
            fileFormat : string, default ".csv.gz"
                string containg the format of the data files downloaded

		Returns
		_ _ _ _ _ _ _ _ _ _ 
							
			None (data are saved in a csv file)"""

    # Iterate over needed years and months
    for y in yearsAndMonths.keys():
        for m in yearsAndMonths[y]:
            # The months are processed differently when they are < 10 (a 0 is added before)
            	if (m > 9):
                    # Download the archive file and save it into the output_directory
                    pathAndFileArch, headNotUse = urllib.urlretrieve(url+baseName+str(y)+str(m)+fileFormat,\
                                                                     output_directory+baseName+\
                                                                     str(y)+str(m)+fileFormat)
                else:
                    pathAndFileArch, headNotUse = urllib.urlretrieve(url+baseName+str(y)+"0"+str(m)+fileFormat,\
                                                                     output_directory+baseName+\
                                                                     str(y)+"0"+str(m)+fileFormat)
                
                # Decompress the downloaded zip into the output_directory
                str_object1 = open(pathAndFileArch, 'rb').read()
                str_object2 = zlib.decompress(str_object1, zlib.MAX_WBITS|32)
                f = open(pathAndFileArch[0:-len(fileFormat)]+".csv", 'wb')
                f.write(str_object2)
                f.close()
            
            
def loadMeteoFranceSynop(cityCode = 7222,
                         yearsAndMonths = {2018: range(1, 13), 2019: range(1, 10)},
                         inputDirectory = "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP/",
                         baseName = "synop.",
                         saveFile = False,
                         output_directory = "/home/decide/Data/Climato/Donnees_compilees/MeteoFrance/DonneesLibres/SYNOP/"):
    """ Load local meteorological data of one station that has been
    previously downloaded on SYNOP Meteo-France web site:
    https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32
    The data may be saved on the computer.

    	Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
        cityCode : integer, default 7222 (Nantes)
            integer corresponding to the code of the station to download 
            (this information can be found either on the Météo-France web-site -
            https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32
            or either in the "./CreateWeatherAlert/Ressources/ListeStationsSynop.csv" file)
        yearsAndMonths : dictionary, default {2018: range(1, 13), 2019: range(1, 10)}
    			dictionary containing as keys the years and as values the corresponding
            month to load
        inputDirectory : String, default "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP/"
            local directory where the downloaded MeteoFrance data should have 
            previously been saved
        baseName : string, default "synop."
            string being part of the base name of the file to load
		saveFile : boolean, default False
			whether or not the output file should be saved on the computer
        output_directory : string, default "/home/decide/Data/Climato/Donnees_compilees/MeteoFrance/DonneesLibres/SYNOP/"
            path where should be saved the output file (if saveFile is True)

	Returns
		_ _ _ _ _ _ _ _ _ _ 
							
		DataFrame containing all meteorological data for the corresponding
        station and the needed date range"""
            
    # Convert months to string
    yearsAndMonthsStr = {}
    for y in yearsAndMonths.keys():
        yearsAndMonthsStr[str(y)] = []
        for m in yearsAndMonths[y]:
            # The months are processed differently when they are < 10 (a 0 is added before)
            if (m <= 9):
                yearsAndMonthsStr[str(y)].append("0" + str(m))
            else:
                yearsAndMonthsStr[str(y)].append(str(m))

    df_W = pd.concat([pd.read_csv(inputDirectory + baseName + y + m + ".csv", \
                                  parse_dates = [1], header = 0, sep = ";",\
                                  index_col = None, na_values = 'mq') for y in yearsAndMonthsStr.keys()
                                                 for m in yearsAndMonthsStr[y]])
    
    # Recover the data corresponding to the city code and use datetime object for indexing
    df_output = df_W[df_W.numer_sta == 7222]
    df_output.loc[:,"date"] = pd.to_datetime(df_output["date"], format = "%Y%m%d%H%M%S")
    df_output.index = df_output["date"]
    df_output.sort_index().drop(["date", "numer_sta"], axis = 1, inplace = True)
    
    if saveFile == True:
        df_output.to_csv(output_directory+cityCode+".csv")
        
    return df_output