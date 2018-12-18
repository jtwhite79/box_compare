       program finaltime

       implicit none

       double precision time, easting
       character*15   atemp
       character*200  infile,outfile
       character*200  cline,dline

       open(unit=*,action='read',carriagecontrol='list')

100    write(6,110,advance='no')
110    format(' Enter name of ADV output file: ')
       read(5,'(a)') infile
       open(unit=10,file=infile,status='old',err=100)
       
       do
         read(10,'(a)',end=9000) cline
         if(index(cline(1:20),'PROJECTED').ne.0) go to 150
         dline=cline
       end do

150    continue
       close(unit=10)
       atemp=dline(135:147)
       read(atemp,'(f15.0)',err=9100) time
       atemp=dline(37:49)
       read(atemp,'(f15.0)',err=9200) easting
       write(6,160) trim(infile)
160    format(' - file ',a,' read ok.')       


200    write(6,210,advance='no')
210    format(' Enter name for this program''s output file: ')
       read(5,'(a)') outfile
       if(outfile.eq.' ') go to 200
       open(unit=20,file=outfile)
       write(20,230) time
230    format(' Time of particle emergence',t40,': ',1pg14.7)
       write(20,240) easting
240    format(' Easting of particle emergence',t40': ',1pg14.7)
       write(6,250) trim(outfile)
250    format(' - file ',a,' written ok.')       

       go to 9999

9000   write(6,9010)
9010   format(' Error: cannot find "PROJECTED" string.')
       stop
9100   write(6,9110)
9110   format(' Error: cannot read exit time from ADV output ',
     + 'file.')
       stop
9200   write(6,9210)
9210   format(' Error: cannot read exit easting from ADV output ',
     + 'file.')
       stop


9999   continue

       end
