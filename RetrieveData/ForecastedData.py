#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:54:28 2019

@author: Jérémy Bernard
"""

import urllib
import zlib

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
							
					An "optimal" fitted statsmodels linear model with an intercept selected by forward selection evaluated by adjusted R-squared"""

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
                f = open(pathAndFileArch[0:-len(fileFormat)], 'wb')
                f.write(str_object2)
                f.close()
            
