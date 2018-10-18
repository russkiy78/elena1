#!/usr/bin/python 

#  Description: POST PROCESSING OF THE RAW DATASET: IRGASON (OS VERSION EC100.07.01), OUTPUT VIA RS485 PORT  
#  Experiment: ANTARCTICA, 2018
#  Current version: 0.01
#  
#  

# ################## GENERAL COMMENTS #############################################################
# Output data from the Irgason are collected  and stored in the "in-house" developed datalogger
# the data loger is based on MOXA computer (https://www.moxa.com/product/IA260.htm).
# It has a light version of linux as well as shell scripts combining the outputs from GPS and EC100,
# and run the data saving process.
# the final data is packed on bz2 file once a day.


# #################### Libraries  ######################################################
# import os
import bz2
import numpy
from datetime import datetime

# ################### CONSTANTS #########################################################
MU_WPL = 28.97 / 18.02
R = 8.3143e-3  # kPa m3 K-1 mol-1
RD = R / 28.97  # kPa m3 K-1 g-1
RV = R / 18.02  # kPa m3 K-1 g-1
sonic_azi = 137.
SIG_STRG = 0.7
von_karman = 0.4
Zm = 2.0  # m
gravity = 9.80665  # m s-2
Lk = 2.0  # com o valor b == 3.7 (pag 516 klunj) L'~2.0 (figure A1)
Ac = 4.28
Ad = 1.68
Bk = 3.42
alpha1 = -0.8
hd = 24.  # 24 horas
XCO2 = 44.0095  # g mol-1
XCO2_mg = 44009.5  # mg mol-1


# ################ Working directory ########################################################################


def get_from_file(filename):
    structure = []
    with bz2.BZ2File(filename, "r") as fobj:
        for line in fobj:
            line = line.decode("utf-8").strip().split(',')
            firsfline = line[0].split(' ')
            structure.append({
                'DateStamp': datetime.strptime(firsfline[0] + firsfline[1], '%y%m%d%H%M%S%Z'),
                'Ux': float(firsfline[2]),
                'Uy': float(line[1]),
                'Uz': float(line[2]),
                'SonicTemperature': float(line[3]),
                'SonicDiagnosticFlag': float(line[4]),
                'CO2Density': float(line[5]),
                'H2ODensity': float(line[6]),
                'GasDiagnosticFlag': float(line[7]),
                'AirTemperature': float(line[8]),
                'AirPressure': float(line[9]),
                'CO2SignalStrengthNominally': float(line[10]),
                'H2OSignalStrengthNominally': float(line[11]),
                'PressureDifferential': float(line[12]),
                'SourceHousingTemperature': float(line[13]),
                'DetectorHousingTemperature': float(line[14]),
                'CounterArbitrary': int(line[15]),
                'SignatureArbitrary': line[16]
            })
        return structure

def despikes (data, num_stdev,nmh, name_out):
    co2_final = data
    nn_co2 = num_stdev
    media_co2 = numpy.mean(data)

    # variable stats_co2 never used in this function!
    # stats_co2 = moment(data, sdev=sigma_co2)
    # variable sigma_co2 is implicitly come from outer space!


    # med_mov = TS_SMOOTH(data, 6000, / DOUBLE)


despikes (1,2,3,4)

# print(get_from_file("IRGASON-180201_0000.dat.bz2")[0])
