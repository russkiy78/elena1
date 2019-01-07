# Code to perform post processing procedure for Irgason measurement system: https://www.campbellsci.com/irgason
# OS EC100.07.01 (RS485) and OS EC100.08.01 (RS485)
# examples of output from the system are
# IRGASON-180201_0000.dat.bz2 (EC100.07.01): 15 column reports Pressure diffirential (see Irgason_output.pdf)
# IRGASON-181012_0000.dat.bz2 (EC100.08.01): 15 column reports spectroscopically corrected CO2 flux (Ivan Bogoev)
# raw data can be preformed on 10 and 20 Hz temporal resolution
# original code to perform post processing for the OS EC100.07.01 is written and commented by Miguel Potes (IDL): see code by Miguel.txt

# In main.py
# Code structure

# Data filtering: 
The fluxes were calculated withing 30 minute intervals (INTERVAL) using the raw data filtered within two steps. On the first step, the time intervals covered by less than 50 % of total measurements (FILTERING:BY A NUMBER OF MEASUREMENT) were excluded, and then the flagging procedure was applied (FILTERING: BY DAGNOSTIC FLAG). In the flagging, we excluded the raw data with non zero values in the analysers’ flags (SonicDiagnosticFlag and GasDiagnosticFlag) and then the raw data with the CO2 and H2O signal strength less then 0.7. The data quality control was performed by a number of the intervals rejected (RN), and a percentage of measurements accepted within each time interval (AN, %) on the fist step of the filtering. Further calculations of the fluxes were done for the intervals with AN more than 25 %. On the second step, the raw data were processed to remove spikes (Vickers and Mahrt, 1997). In the despiking procedure the standard deviation (STD) was calculated with ignoring the non-valid data values within the moving window (DESPIKING_MA_PERIOD=60). The length of the DESPIKING_MA_PERIOD is defined depending on the frequency of the raw data. The despiking thresholds (DESPIKING_THRESHOLD) were set as 3.5 of for the wind Ux-, Uy- components, the sonic temperature, CO2 and H2O densities, the air temperature and pressure, and 5.0 of the SD for the wind Uz- component (DESPIKING_VALUES). The maximum number of the spikes a row that are counted as spikes (DESPIKING_MAX_IN_ROW) was set to 10 instead of 3 according to Vickers and Mahrt (1997). The spikes detected were replaced by the mean values within the moving window. The despiking procedure was repeated up to 20 times, or until no more spikes are found. The spikes detected (SN) were counted for the quality control on the second step of the filtering. 
