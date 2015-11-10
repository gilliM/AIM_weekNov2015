import sys

# python script to merge several xmf files into a Time Collection

# pass a list of files
files = sys.argv[2:]
Nfiles = len(files)
fout = open(sys.argv[1],'w')

# write the first couple of lines
fout.write('<?xml version="1.0" ?>\n')
fout.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
fout.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
fout.write('  <Domain>\n')
fout.write('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
inrange=0
time=0
for f in files:
    print f
    fout.write('\n')

    # open the input file
    fin = open(f,'r')
    if(fin<0):
        print 'File '+ f +' does not exist'
        continue

    # read the input file from the open grid to close grid
    # and write it to the output file
    while not inrange:
        line=fin.readline()
        if(line.find('Grid')>=0):
            fout.write(line)
            inrange=1

    line=fin.readline()
    fout.write('<Time Value="%05d"/>\n' % (time))
    time += 1

    while inrange:
        line=fin.readline()
        fout.write(line)
        if(line.find('/Grid')>=0):
            inrange=0
    fin.close()

fout.write('\n')
fout.write('    </Grid>\n')
fout.write('  </Domain>\n')
fout.write('</Xdmf>\n')

fout.close()
