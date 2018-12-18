@echo off
Rem ###################################
Rem Some intermediate files are deleted.
Rem ###################################

del heads.dat > nul

Rem ###################################
Rem Now the actual model is run
Rem ###################################

mf2k < mf2k.in 
finaltime < finaltime.in 
mod2obs < mod2obs.in 
