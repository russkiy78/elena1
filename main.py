#!/usr/bin/python 

#  Description: POST PROCESSING OF THE RAW DATASET: IRGASON (OS VERSION EC100.07.01), OUTPUT VIA RS485 PORT  
#  Experiment: ANTARCTICA, 2018
#  Current version: 0.01
#  
#  

################### GENERAL COMMENTS ########################################################################
#Output data from the Irgason are collected  and stored in the "in-house" developed datalogger 
#the data loger is based on MOXA computer (https://www.moxa.com/product/IA260.htm). 
#It has a light version of linux as well as shell scripts combining the outputs from GPS and EC100, and run the data saving process.
#the final data is packed on bz2 file once a day.


##################### Libraries  ######################################################
import os
import bz2
from datetime import datetime

#################### CONSTANTS #########################################################
MU_WPL=28.97/18.02
R=8.3143e-3	 ########### kPa m3 K-1 mol-1
RD=R/28.97	 ########### kPa m3 K-1 g-1
RV=R/18.02	 ########### kPa m3 K-1 g-1
sonic_azi=137.
SIG_STRG=0.7
von_karman=0.4
Zm=2.0        	 ########### m
gravity=9.80665	 ########### m s-2
Lk=2.0           ########### com o valor b == 3.7 (pag 516 klunj) L'~2.0 (figure A1)
Ac=4.28
Ad=1.68
Bk=3.42
alpha1=-0.8
hd=24.           ######## 24 horas
XCO2=44.0095     ######## g mol-1
XCO2_mg=44009.5  #######mg mol-1

################# Working directory ########################################################################

path = os.getcwd()
filename = path+"/"+"IRGASON-180201_0000.dat.bz2"
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
    print(structure[1])

#################################################################################################################
############################# FUNCTIONS #########################################################################

def DESPIKES (data, num_stdev,nmh, name_out):

	co2_final=data
	nn_co2=num_stdev
	media_co2=MEAN(data,/NAN)
	stats_co2=moment(data,sdev=sigma_co2)
	med_mov=TS_SMOOTH(data,6000,/DOUBLE)

    ### poe os 0's a NAN
	if n_elements(where(co2_final eq 0.)) GT 1 THEN co2_final[where(co2_final EQ 0.)]=!VALUES.D_NAN

	ns_co2=0

	spikes_co2=co2_final gt med_mov+nn_co2*sigma_co2 or co2_final lt med_mov-nn_co2*sigma_co2

	### fazer o caso de haver spike na posicao 0 e na posicao 35999
	IF spikes_co2(0) EQ 1  then co2_final(0)=!VALUES.D_NAN
	IF spikes_co2(n_elements(spikes_co2)-1) EQ 1 then co2_final(n_elements(spikes_co2)-1)=!VALUES.D_NAN

	### tratar os spikes CO2, H2O, X, Y
	### para o caso de varios spikes seguidos são tratados em várias vezes graças ao while
	while n_elements(where(spikes_co2)) GT 1 do begin

		### array com localização dos spikes (vai sendo renovado a cada while)
		id_sp=where(spikes_co2)

			FOR kk=0, n_elements(where(spikes_co2))-1 do begin

				if id_sp(kk) gt 0 and id_sp(kk) lt n_elements(co2_final)-1 then begin
					### se for absurdo por NAN
					if co2_final(id_sp(kk)) GT 1.03*media_co2 or co2_final(id_sp(kk)) LT .97*media_co2 then $
					co2_final(id_sp(kk))=!VALUES.D_NAN $

					### substitui o spike pela media dos vizinhos
					else co2_final(id_sp(kk))=(co2_final(id_sp(kk)-1)+co2_final(id_sp(kk)+1))/2.
				endif
			endfor
			ns_co2++

		### faz novamente as contas da media, sigma e media movel

		spikes_co2=co2_final gt med_mov+nn_co2*sigma_co2 or co2_final lt med_mov-nn_co2*sigma_co2
		id_sp2=where(spikes_co2)
		if n_elements(id_sp2) ge n_elements(id_sp) then spikes_co2=-1
	endwhile

	IF n_elements(where(spikes_co2)) EQ 1 THEN begin

			id_sp=where(spikes_co2)

			### para o caso de 1 spike
			IF id_sp NE -1 THEN begin
				co2_final(id_sp)=(co2_final(id_sp-1)+co2_final(id_sp+1))/2.
			ENDIF
			ns_co2++

	ENDIF

return co2_final
