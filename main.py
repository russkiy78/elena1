#!/usr/bin/python 

# import os
import bz2
import math
import numpy
from datetime import datetime, timedelta
import time
start_time = time.time()

# ###########################CONFIGURATION OF THE EXPERIMENT#################################
FILE = "IRGASON-181012_1229.dat.bz2"  # relative path to file 
FORMAT = "08"  # "08"|"07"            # OS EC100 08.01 or EC100 07.01
FREQUENCY = 20  # the unpromtem output frequency, Hz
INTERVAL = 30  # interval for the flux calculation, min
Sonic_azimut = 137  # direction of the instriments, degree
Sonic_height = 2  # height of intrument, m

# ###################### PHYSICAL COSTANTS #################################################
R = 8.3144598  # the universal gas constant, kg m2 s−2 K−1 mol−1


# ############################# FUNCTIONS DECLARATION ######################################
# ############################# READ DATA ##################################################

def get_from_file(filename, file_format, freq):
    struct = {
        "Format": file_format,
        "Frequency": freq,
        "From": datetime.max,
        "To": datetime.min,
        "RawData": []}
    with bz2.BZ2File(filename, "r") as fobj:
        counter = 0
        for line in fobj:
            line = line.decode("utf-8").strip().split(',')
            firstfline = line[0].split(' ')
            if len(line) == 17:
                dt = datetime.strptime(firstfline[0] + firstfline[1], '%y%m%d%H%M%S%Z')
                if dt > struct["To"]:
                    struct["To"] = dt
                if dt < struct["From"]:
                    struct["From"] = dt
                struct["RawData"].append({
                    'DateStamp': dt,
                    'Ux': float(firstfline[2]) if firstfline[2] else numpy.nan,  # ms-1
                    'Uy': float(line[1]) if line[1] else numpy.nan,  # ms-1
                    'Uz': float(line[2]) if line[2] else numpy.nan,  # ms-1
                    'SonicTemperature': float(line[3]) if line[3] else numpy.nan,  # C
                    'SonicDiagnosticFlag': float(line[4]) if line[4] else numpy.nan,  #
                    'CO2Density': float(line[5]) if line[5] else numpy.nan,  # mg m-3
                    'H2ODensity': float(line[6]) if line[6] else numpy.nan,  # g m-3
                    'GasDiagnosticFlag': float(line[7]) if line[7] else numpy.nan,  #
                    'AirTemperature': float(line[8]) if line[8] else numpy.nan,  # C
                    'AirPressure': float(line[9]) if line[9] else numpy.nan,  # kPa
                    'CO2SignalStrengthNominally': float(line[10]) if line[10] else numpy.nan,  #
                    'H2OSignalStrengthNominally': float(line[11]) if line[11] else numpy.nan,  #
                    'PressureDifferential': float(line[12]) if line[12] else numpy.nan,  # kPa
                    'CO2Correct': float(line[12]) if line[12] else numpy.nan,  # mg m-3
                    'SourceHousingTemperature': float(line[13]) if line[13] else numpy.nan,  # C not needed
                    'DetectorHousingTemperature': float(line[14]) if line[14] else numpy.nan,  # C not needed
                    'CounterArbitrary': int(line[15]) if line[15] else numpy.nan,  # not needed
                    'SignatureArbitrary': line[16] if line[16] else ''  # not needed
                })
                # for testing only!

                # counter += 1
                # if counter == 100000:
                #   return struct

        return struct


# ########################## SPLITTING INTO 30 MIN INTERVALS ##################################################
def split_struct(struct, interval):
    if len(struct["RawData"]) < 1:
        return False

    data = [struct["From"].replace(hour=00, minute=00, second=00) + timedelta(minutes=i)
            for i in range(0, (24 * 60), interval)]
    struct["Filtered"] = [{"From": data[i], "Data": []} for i in range(len(data))]

    for element in struct["RawData"]:
        index = [i for i in range(len(data)) if
                 data[i] <= element['DateStamp']]
        if len(index) > 0:
            struct["Filtered"][index[-1]]["Data"].append(element)
        else:
            return False
    return struct


# ################################ FILTERING: BY A NUMBER OF MEASUREMENTS #####################################
def filter_struct(struct, interval, freq, threshold=0.5):
    struct = [x for x in struct if len(x["Data"]) >= freq * interval * 60 * threshold]
    return struct


# ############## SONIC TEMPERATURE HUMIDITY CORRECTION: Kaimal and Gaynor (1991) ###############################
def add_t_corrected(x, r):
    x.update({"TemperatureC": (x["SonicTemperature"] + 273.15) /
                              ((1 + 0.32 * x['H2ODensity'] * 1000 * r * (x["SonicTemperature"] + 273.15)) /
                               18.02 * x['AirPressure'] * 1000)
              })
    return x


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


# ######################## MAIN CODE ############################################################################

print("Get from file...")
structure = get_from_file(FILE, FORMAT, FREQUENCY)
print("--- %s seconds ---" % (time.time() - start_time))

print("Add Temperature Corrected")
structure["RawData"] = [add_t_corrected(x, R) for x in structure["RawData"]]
print("--- %s seconds ---" % (time.time() - start_time))

print("Split by %d min intervals" % INTERVAL)
structure = split_struct(structure, INTERVAL)
print("--- %s seconds ---" % (time.time() - start_time))

print(len(structure["Filtered"]))

print("Filter for number of measurement by interval (min 50% for each) ")
structure["Filtered"] = filter_struct(structure["Filtered"], INTERVAL, FREQUENCY, 0.5)
print("--- %s seconds ---" % (time.time() - start_time))

print(len(structure["Filtered"]))

print("Filter for SonicDiagnosticFlag  GasDiagnosticFlag CO2SignalStrengthNominally H2OSignalStrengthNominally")
structure["Filtered"] = diagnostic_filter(structure["Filtered"])
print("--- %s seconds ---" % (time.time() - start_time))

print(structure["Filtered"][0]['From'])
print(len(structure["Filtered"][0]['Data']))

data_q_c = len(structure["Filtered"]) / (24 * 60 / INTERVAL / 100)
print("Data Quality Control = {} % ".format(data_q_c))
