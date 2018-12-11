#!/usr/bin/python 
#  Software description: POST PROCESSING OF THE RAW DATASET: IRGASON (OS VERSION EC100.07.01), OUTPUT VIA RS485 PORT  
#  Experiment: ANTARCTICA, 2018
#  Current version: 0.01
# Contact: elena.shevnina@fmi.fi 

################### GENERAL COMMENTS ########################################################################
# Output data from the Irgason are collected  and stored in the "in-house" developed datalogger
# the data loger is based on MOXA computer (https://www.moxa.com/product/IA260.htm).
# It has a light version of linux as well as shell scripts combining the outputs from GPS and EC100, and run the data saving process.
# the final data is packed on bz2 file once a day.


##################### Libraries  ######################################################
import os
import bz2
import math
import matplotlib.pyplot as plt
import numpy
from datetime import datetime, timedelta
import time

start_time = time.time()

# ###########################CONFIGURATION OF THE EXPERIMENT#################################
# FILE = "IRGASON-181012_1229.dat.bz2"     # relative path to file
FILE = "IRGASON-180201_0000.dat.bz2"
DRAW_PATH = '/home/russkiy/elenagraph/'  # relative path to the directory with graphs
FORMAT = "07"  # "08"|"07"                 # OS EC100 08.01 or EC100 07.01
FREQUENCY = 10  # the unpromtem output frequency, Hz
INTERVAL = 30  # interval for the flux calculation, min
SONIC_AZIMUT = 137  # direction of the instruments, degree
SONIC_HEIGHT = 2  # height of instrument, m

# ###################### PHYSICAL CONSTANTS #################################################
PHYS_R = 8.3144598  # the universal gas constant, Pa m3 K-1 mol-1             //Pa =  kg⋅m−1⋅s−2
PHYS_DA = 287.058  # the specific gas constant for dry air, Pa m3 K-1 kg-1
PHYS_WV = 461.5  # the specific gas constant for water vapor, Pa m3 K-1 Kg-1
PHYS_HDA = 1004.67  # the specific heat of dry air at a constant pressure, J kg-1 K-1 // J =  kg⋅m2⋅s−2
PHYS_LHV = 2501000  # the latent heat of vaporization at 0ºC, J kg-1
PHYS_W = 0.018016  # the molar mass of water vapor, kg mol-1
PHYS_A = 0.02897  # the molar mass of dry air, kg mol-1
PHYS_G = 9.80665  # the gravity accelation, m s-2

XCO2 = 44.0095  # g mol-1

# ##################### EMPIRICAL CONSTANTS #########################################
Lk = 2.0  # acc. Klunj, 2004: L'~2.0 (figure A1)
bK = 3.7  # acc. Klunj, 2004
Ac = 4.28
Ad = 1.68
Bk = 3.42
alpha1 = -0.8

# ###################### DESPIKING PARAMETERS #################################################
DESPIKING_VALUES = ['Ux', 'Uy', 'Uz', 'SonicTemperature', 'CorrectedTemperature', 'CO2Density', 'H2ODensity']
DESPIKING_THRESHOLD = [3.5, 3.5, 5, 3.5, 3.5, 3.5, 3.5]
DESPIKING_MAXCIRCLES = 20
DESPIKING_MAX_IN_ROW = 10  # maximum spikes in a row that are counted as spikes, should be 3 according to Vickers and Mahrt, 1997
DESPIKING_MA_PERIOD = 60  # moving window length, calculated as FREQUENCY, Hz * time_period, sec (10 Hz *6 sec)

# ###################### !!!DEBUG CONSTANTS (FOR DEBUGGING ONLY)!!! ############################
DEBUG_MAX_INTERVALS = 2  # the number of intervals (received from the file) MUST BE 0 FOR PRODUCTION

# ###################### DRAWING CONSTANTS  ############################
DRAW_DPI = 400
DRAW_FORMAT = 'png'
DRAW_SAVE_TO_FILE = False  # save graphic to dist instead of show
DRAW_FILENAME_TEMPLATE = "%Y-%m-%d %H:%M:%S"

# DRAW_WIDTH_MAIN = 0.5
DRAW_COLOR_MAIN = 'black'

DRAW_WIDTH_MEAN = 0.2
DRAW_COLOR_MEAN = 'green'

DRAW_WIDTH_MA = 0.2
DRAW_COLOR_MA = 'red'

DRAW_WIDTH_SDMA = 0.2
DRAW_COLOR_SDMA = 'blue'


# ############################# FUNCTIONS DECLARATION ######################################
# ############################# READ DATA + SPLIT ##################################################

def get_from_file(filename, file_format, freq, interval, interval_count=0):
    struct = {
        "Format": file_format,
        "Frequency": freq,
        "Intervals": interval,
        "Day": False,
        "Data": []}
    with bz2.BZ2File(filename, "r") as fobj:

        day_intervals = []
        interval_counter = 0
        prev_interval = -1

        for line in fobj:
            line = line.decode("utf-8").strip().split(',')
            firstfline = line[0].split(' ')
            if len(line) == 17:
                # get time of record
                dt = datetime.strptime(firstfline[0] + firstfline[1], '%y%m%d%H%M%S%Z')

                #  create structure for first time
                if not struct["Day"]:
                    struct["Day"] = dt.replace(hour=00, minute=00, second=00)

                    day_intervals = [struct["Day"] + timedelta(minutes=i)
                                     for i in range(0, (24 * 60), interval)]

                    struct["Data"] = [{"From": day_intervals[i], "Data": []} for i in range(len(day_intervals))]

                index = [i for i in range(len(day_intervals)) if day_intervals[i] <= dt]

                if len(index) > 0:

                    if index[-1] != prev_interval:
                        interval_counter += 1
                        prev_interval = index[-1]

                    if interval_counter > interval_count > 0:
                        return struct

                    struct["Data"][index[-1]]["Data"].append({
                        'DateStamp': dt,
                        'Ux': float(firstfline[2]) if firstfline[2] else numpy.nan,  # wind x-component, m s-1
                        'Uy': float(line[1]) if line[1] else numpy.nan,  # wind y-component, m s-1
                        'Uz': float(line[2]) if line[2] else numpy.nan,  # wind z-component, m s-1
                        'SonicTemperature': float(line[3]) if line[3] else numpy.nan,  # degree Celcium
                        'SonicDiagnosticFlag': float(line[4]) if line[4] else numpy.nan,  # flag, non unit
                        'CO2Density': float(line[5]) if line[5] else numpy.nan,  # density, !!!mg m-3
                        'H2ODensity': float(line[6]) if line[6] else numpy.nan,  # density, g m-3
                        'GasDiagnosticFlag': float(line[7]) if line[7] else numpy.nan,  # flag, non unit
                        'AirTemperature': float(line[8]) if line[8] else numpy.nan,  # degree Celcium
                        'AirPressure': float(line[9]) if line[9] else numpy.nan,  # pressure, !!!kPa
                        'CO2SignalStrengthNominally': float(line[10]) if line[10] else numpy.nan,  # fraction
                        'H2OSignalStrengthNominally': float(line[11]) if line[11] else numpy.nan,  # fraction
                        'PressureDifferential': float(line[12]) if line[12] else numpy.nan,  # pressure, !!!kPa
                        'CO2Correct': float(line[12]) if line[12] else numpy.nan,  # density, !!!mg m-3
                        'SourceHousingTemperature': float(line[13]) if line[13] else numpy.nan,
                        # degree Celcium, not used
                        'DetectorHousingTemperature': float(line[14]) if line[14] else numpy.nan,
                        # degree Celcium, not used
                        'CounterArbitrary': int(line[15]) if line[15] else numpy.nan,  # not used
                        'SignatureArbitrary': line[16] if line[16] else ''  # not used
                    })

        return struct


# ############## SONIC TEMPERATURE HUMIDITY CORRECTION: Kaimal and Gaynor (1991) ###############################
def add_t_corrected(x):
    for i in range(len(x['Data'])):

        tc = numpy.nan

        if x['Data'][i]["H2ODensity"] > 0 and x['Data'][i]['AirPressure'] > 0:
            # suggestion by M. Potes, 05.12.2018
            tc = (x['Data'][i]["SonicTemperature"] + 273.15) / (
                    (1 + 0.32 * x['Data'][i]['H2ODensity'] * (PHYS_R / 1000) * (
                            x['Data'][i]["SonicTemperature"] + 273.15)) / (
                            (PHYS_W * 1000) * x['Data'][i]['AirPressure']))
            # !!! here H2ODensity in g m-3
            # !!! Air pressure in kPa

        x['Data'][i].update({"CorrectedTemperature": tc})
    return x


# ################################ FILTERING: BY A NUMBER OF MEASUREMENTS #####################################
def filter_struct(struct, interval, freq, threshold=0.5):
    struct = [x for x in struct if len(x["Data"]) >= freq * interval * 60 * threshold]
    return struct


# ########################## FILTERING: BY DAGNOSTIC FLAGS ####################################################
def diagnostic_filter(struct):
    for index in range(len(struct)):
        struct[index]["Data"] = [x for x in struct[index]["Data"]
                                 if x['SonicDiagnosticFlag'] == 0
                                 and x['GasDiagnosticFlag'] == 0
                                 and x['CO2SignalStrengthNominally'] >= 0.7
                                 and x['H2OSignalStrengthNominally'] >= 0.7
                                 ]
    return struct


def drawplt(struct):
    for i in range(len(struct)):

        for value in range(len(DESPIKING_VALUES)):
            print(DESPIKING_VALUES[value])

            math_mass = numpy.array([x[DESPIKING_VALUES[value]] for x in struct[i]['Data']])
            mean = numpy.mean(math_mass)
            std = numpy.nanstd(math_mass)

            ma = moving_average(math_mass, DESPIKING_MA_PERIOD)
            plt.clf()
            plt.plot(numpy.full(len(math_mass), mean), color=DRAW_COLOR_MEAN, linewidth=DRAW_WIDTH_MEAN)
            plt.plot(math_mass, ',', color=DRAW_COLOR_MAIN)
            plt.plot(ma, color=DRAW_COLOR_MA, linewidth=DRAW_WIDTH_MA)
            plt.plot(ma + float(std * DESPIKING_THRESHOLD[value]), color=DRAW_COLOR_SDMA, linewidth=DRAW_WIDTH_SDMA)
            plt.plot(ma - float(std * DESPIKING_THRESHOLD[value]), color=DRAW_COLOR_SDMA, linewidth=DRAW_WIDTH_SDMA)

            plt.title(
                '{} {} (spikes {})'.format(struct[i]['From'].strftime(DRAW_FILENAME_TEMPLATE),
                                           DESPIKING_VALUES[value],
                                           struct[i]['Spikes'][DESPIKING_VALUES[value]]))
            if DRAW_SAVE_TO_FILE:
                plt.savefig(
                    '{}{}-{}.{}'.format(DRAW_PATH, struct[i]['From'].strftime(DRAW_FILENAME_TEMPLATE),
                                        DESPIKING_VALUES[value], DRAW_FORMAT),
                    format=DRAW_FORMAT, dpi=DRAW_DPI)
            else:
                plt.show()


def get_interpolate(mass, index):
    if index == len(mass) - 1:
        return (mass[-2] + mass[-1]) / 2
    elif index == 0:
        return (mass[0] + mass[1]) / 2
    else:
        return (mass[index - 1] + mass[index + 1]) / 2


def despiking(struct):
    for i in range(len(struct)):

        struct[i]['Spikes'] = {}

        for value in range(len(DESPIKING_VALUES)):

            print(DESPIKING_VALUES[value])

            struct[i]['Spikes'][DESPIKING_VALUES[value]] = 0
            math_mass = numpy.array([x[DESPIKING_VALUES[value]] for x in struct[i]['Data']])

            for despiking_index in range(DESPIKING_MAXCIRCLES):
                no_spike = True

                # get moving average
                ma = moving_average(math_mass, DESPIKING_MA_PERIOD)

                # get standart deviation, ignoring NAN values
                std = numpy.nanstd(math_mass)

                dindex = 0

                while dindex < len(math_mass):

                    spike_found = 0
                    # checking for spike
                    while dindex + spike_found < len(math_mass) and \
                            (math_mass[dindex + spike_found] > ma[dindex + spike_found] +
                             std * DESPIKING_THRESHOLD[value] or
                             math_mass[dindex + spike_found] < ma[dindex + spike_found] -
                             std * DESPIKING_THRESHOLD[value]):
                        spike_found += 1

                    if 0 < spike_found <= DESPIKING_MAX_IN_ROW:

                        # add one spike
                        if despiking_index == 0:
                            struct[i]['Spikes'][DESPIKING_VALUES[value]] += 1
                        no_spike = False

                        # replace spikes with mean values
                        for spike_i in range(dindex, dindex + spike_found):
                            math_mass[spike_i] = ma[spike_i]

                    dindex += (spike_found + 1)

                # no spikes at this round -  all done, break the circle
                if no_spike:
                    break

            # write to Data structure
            for x in range(len(struct[i]['Data'])):
                struct[i]['Data'][x][DESPIKING_VALUES[value]] = math_mass[x]
    return struct


def moving_average(y, n):
    y_padded = numpy.pad(y, (n // 2, n - 1 - n // 2), mode='edge')
    return numpy.convolve(y_padded, numpy.ones((n,)) / n, mode='valid')


# ####################### to calculate the density of dry air, kg m-3 ##########################################
# !!! see eq. 2

# ####################### to calculate the air density, kg m-3 ##################################################
# !!! see eq. 3

# ####################### to calculate the Sigma ratio according to Webb, 1980 ##################################
# !!! see eq. 4

# ####################### to calculate the latent heat of vaporization, J kg-1 according to Stull, 1989 #########
# !!! see eq. 5

# ####################### to calculate the specific heat of air, J kg-1 K-1 according to Stull, 1989 ############
# !!! see eq. 6

# ######################## MAIN CODE ############################################################################

print("Get from file...")
structure = get_from_file(FILE, FORMAT, FREQUENCY, INTERVAL, DEBUG_MAX_INTERVALS)
print("--- %s seconds ---" % (time.time() - start_time))

# DELETE EMPTY INTERVALS

structure['Data'] = [i for i in structure['Data'] if len(i['Data']) > 0]

print("Add Temperature Corrected")
structure["Data"] = [add_t_corrected(x) for x in structure["Data"]]
print("--- %s seconds ---" % (time.time() - start_time))

print("Filter for number of measurement by interval (min 50% for each) ")
structure["Data"] = filter_struct(structure["Data"], INTERVAL, FREQUENCY, 0.5)
print("--- %s seconds ---" % (time.time() - start_time))

print("Filter for SonicDiagnosticFlag  GasDiagnosticFlag CO2SignalStrengthNominally H2OSignalStrengthNominally")
structure["Data"] = diagnostic_filter(structure["Data"])
print("--- %s seconds ---" % (time.time() - start_time))

structure["Data"] = despiking(structure["Data"])
print("--- %s seconds ---" % (time.time() - start_time))

for i in range(len(structure["Data"])):
    for value in range(len(DESPIKING_VALUES)):
        print('{} {} spikes {}'.format(structure["Data"][i]["From"],
                                       DESPIKING_VALUES[value],
                                       structure["Data"][i]['Spikes'][DESPIKING_VALUES[value]]))

drawplt(structure["Data"])

'''
x=[]
y=[]

for i in structure['Data']:
    for j in i['Data']:
        y.append(j['Ux'])
        x.append(j['DateStamp'])

plt.plot(x, y)
plt.show()
'''

'''print("Add Temperature Corrected")
structure["RawData"] = [add_t_corrected(x, R) for x in structure["RawData"]]
print("--- %s seconds ---" % (time.time() - start_time))

print("Split by %d min intervals" % INTERVAL)
structure = split_struct(structure, INTERVAL)
print("--- %s seconds ---" % (time.time() - start_time))

print("Filter for number of measurement by interval (min 50% for each) ")
structure["Filtered"] = filter_struct(structure["Filtered"], INTERVAL, FREQUENCY, 0.5)
print("--- %s seconds ---" % (time.time() - start_time))

print("Filter for SonicDiagnosticFlag  GasDiagnosticFlag CO2SignalStrengthNominally H2OSignalStrengthNominally")
structure["Filtered"] = diagnostic_filter(structure["Filtered"])
print("--- %s seconds ---" % (time.time() - start_time))

structure["Filtered"] = despiking(structure["Filtered"])
print("--- %s seconds ---" % (time.time() - start_time))

# print(structure["Filtered"][0]['From'])
# print(len(structure["Filtered"][0]['Data']))
# print(structure["Filtered"][0]['Data'][10])

data_q_c = len(structure["Filtered"]) / (24 * 60 / INTERVAL / 100)
print("Data Quality Control = {} % ".format(data_q_c))
'''
