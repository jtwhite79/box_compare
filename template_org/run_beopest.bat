mkdir worker1
mkdir worker2
cd worker1
copy ..\*.*
start beopest64 temp2_svda /h %COMPUTERNAME%:4004
cd ..
cd worker2
copy ..\*.*
start beopest64 temp2_svda /h %COMPUTERNAME%:4004
cd ..
start beopest64 temp2_svda /h %COMPUTERNAME%:4004
beopest64 temp2_svda /h :4004