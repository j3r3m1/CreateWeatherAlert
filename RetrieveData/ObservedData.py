#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:54:28 2019

@author: Jérémy Bernard
"""

import wget
import zlib

def RetrieveMeteoFranceSynop(yearsList = [2018, 2019],
                            monthsList = range(1, 13),
                            output_directory = "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP",
                            url="https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/",
                            baseName="synop.",
                            fileFormat=".csv.gz"):
    """ Download the data available on the Meteo-France web-site :
        https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32
        The data are recovered and saved in monthly files.

        	Parameters
		_ _ _ _ _ _ _ _ _ _ 
							
			yearsList : list, default [2018, 2019]
				list containing all the years to download
			monthsList : list, default range(1, 13)
				list containing all the months to download
			output_directory : string, default "/home/decide/Data/Climato/Donnees_brutes/MF/Donnees_libres/SYNOP"
				string where is stored the base of the URL construction used
                to download the data
            baseName : string, default "synop."
                string being part of the URL base name to download the data
            fileFormat : string, default ".csv.gz"
                string containg the format of the data files downloaded

		Returns
		_ _ _ _ _ _ _ _ _ _ 
							
					An "optimal" fitted statsmodels linear model with an intercept selected by forward selection evaluated by adjusted R-squared"""

    # Iterate over needed years and months
    for y in yearsList:
        for m in monthsList:
            # The months are processed differently when they are < 10 (a 0 is added before)
            	if (m > 9):
                # Download the file and save it into the output_directory
                fileName = wget.download(url+baseName+str(y)+str(m),\
                                         out = output_directory)
                # Decompress the downloaded zip into the output_directory
                zlib.decompress(fileName)
            else:
                fileName = wget.download(url+baseName+str(y)+"0"+str(m),\
                                         out = output_directory)
                zlib.decompress(fileName)
            
