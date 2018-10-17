import bz2
from datetime import datetime

filename = "IRGASON-180201_0000.dat.bz2"
structure = []

with bz2.BZ2File(filename, "r") as fobj:
    for line in fobj:
        line = line.decode("utf-8").strip().split(',')
        firsfLine = line[0].split(' ')
        structure.append({
            'DateStamp': datetime.strptime(firsfLine[0] + firsfLine[1], '%y%m%d%H%M%S%Z'),
            'Ux': float(firsfLine[2]),
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
    print(structure)
