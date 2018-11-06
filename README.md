# Code to perform post processing procedure for Irgason measurement system: https://www.campbellsci.com/irgason
# OS EC100.07.01 (RS485) and OS EC100.08.01 (RS485)
# examples of output from the system are
# IRGASON-180201_0000.dat.bz2 (EC100.07.01)
# IRGASON-181012_0000.dat.bz2 (EC100.08.01)
# original code to perform post processing for the OS EC100.07.01 is written and commented by Miguel Potes (IDL): see code by Miguel.txt
# the post processing procedure includes following stepts: data reading, data filtering, flux calulations, after flux calculations and the result outputing.
# on the data reading step: 
