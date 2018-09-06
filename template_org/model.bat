@echo off
Rem ###################################
Rem Some intermediate files are deleted.
Rem ###################################

del hk.ref > nul
del heads.dat > nul

Rem ###################################
Rem Now the actual model is run
Rem ###################################

fac2real < fac2real.in > nul
mf2k < mf2k.in > nul
finaltime < finaltime.in > nul
mod2obs < mod2obs.in > nul
