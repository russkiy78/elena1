#!/usr/bin/python 

# import os
import bz2
import math
import numpy
from datetime import datetime, timedelta

# ###########################CONSTANTS#################################
FILE = "IRGASON-181012_1229.dat.bz2"  # relative path
FORMAT = "08"  # "08"|"07"
FREQUENCY = 20
INTERVAL = 30


# ############################################################

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
                    'Ux': float(firstfline[2]) if firstfline[2] else numpy.nan,
                    'Uy': float(line[1]) if line[1] else numpy.nan,
                    'Uz': float(line[2]) if line[2] else numpy.nan,
                    'SonicTemperature': float(line[3]) if line[3] else numpy.nan,
                    'SonicDiagnosticFlag': float(line[4]) if line[4] else numpy.nan,
                    'CO2Density': float(line[5]) if line[5] else numpy.nan,
                    'H2ODensity': float(line[6]) if line[6] else numpy.nan,
                    'GasDiagnosticFlag': float(line[7]) if line[7] else numpy.nan,
                    'AirTemperature': float(line[8]) if line[8] else numpy.nan,
                    'AirPressure': float(line[9]) if line[9] else numpy.nan,
                    'CO2SignalStrengthNominally': float(line[10]) if line[10] else numpy.nan,
                    'H2OSignalStrengthNominally': float(line[11]) if line[11] else numpy.nan,
                    'PressureDifferential': float(line[12]) if line[12] else numpy.nan,
                    'CO2Correct': float(line[12]) if line[12] else numpy.nan,
                    'SourceHousingTemperature': float(line[13]) if line[13] else numpy.nan,
                    'DetectorHousingTemperature': float(line[14]) if line[14] else numpy.nan,
                    'CounterArbitrary': int(line[15]) if line[15] else numpy.nan,
                    'SignatureArbitrary': line[16] if line[16] else ''
                })
                # for testing only!

                #counter += 1
                #if counter == 100000:
                #   return struct

        return struct


def split_struct(struct, interval, freq):
    if len(struct["RawData"]) < 1:
        return False

    data = [struct["From"].replace(hour=00, minute=00, second=00) + timedelta(minutes=i)
            for i in range(0, (24 * 60), interval)]
    struct["Filtered"] = [[] for i in range(len(data))]

    for element in struct["RawData"]:
        index = [i for i in range(len(data)) if
                 data[i] <= element['DateStamp']]
        if len(index) > 0:
            struct["Filtered"][index[-1]].append(element)
        else:
            return False
    return struct


def filter_struct(struct, interval, freq):
    struct["Filtered"] = [x for x in struct["Filtered"] if len(x) >= freq * interval * 60 / 2]
    return struct


def add_t_corrected(x):
    x.update({"TemperatureC": (x["SonicTemperature"] + 273.15) /
                              (1 + 0.32 * x['H2ODensity'] * 8.3143 * 0.001 * (x["SonicTemperature"] + 273.15) /
                               18.02 * x['AirPressure'])
              })
    return x


def diagnostic_filter(struct):
    for index in range(len(struct["Filtered"])):
        struct["Filtered"][index] = [x for x in struct["Filtered"][index]
                                     if x['SonicDiagnosticFlag'] == 0
                                     and x['GasDiagnosticFlag'] == 0
                                     and x['CO2SignalStrengthNominally'] >= 0.7
                                     and x['H2OSignalStrengthNominally'] >= 0.7
                                     ]
    return struct


print("Get from file...")
structure = get_from_file(FILE, FORMAT, FREQUENCY)

print("Add Temperature Corrected")
structure["RawData"] = [add_t_corrected(x) for x in structure["RawData"]]

print("Split by %d min intervals" % INTERVAL)
structure = split_struct(structure, INTERVAL, FREQUENCY)

print("Filter for number of measurement by interval (min 50% for each) ")
structure = filter_struct(structure, INTERVAL, FREQUENCY)

print("Filter for SonicDiagnosticFlag  GasDiagnosticFlag CO2SignalStrengthNominally H2OSignalStrengthNominally")
structure = diagnostic_filter(structure)

print("Filter again for number of measurement by interval (min 50% for each) ")
structure = filter_struct(structure, INTERVAL, FREQUENCY)

# print ([ x['SonicDiagnosticFlag'] for x in structure["RawData"]])

data_q_c = len(structure["Filtered"]) / (24 * 60 / INTERVAL / 100)
print("Data Quality Control = {} % ".format(data_q_c))
