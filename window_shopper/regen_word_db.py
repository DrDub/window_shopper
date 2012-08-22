import bsddb;
import os;
import subprocess;
import sys;


db = bsddb.hashopen(sys.argv[2], 'w')

dump = open(sys.argv[1], 'r')
in_key = True
values = []
count = 0
line_count = 0
SEPARATOR = "+=" * 50 + "\n"
for line in dump:
    line_count = line_count + 1
    if line == SEPARATOR:
        count = count + 1
        db[key] = ''.join(values)
        in_key = True
        values = []
    elif in_key:
        key = line.strip()
        in_key = False
    else:
        values.append(line)
db.close()
print "read", line_count, "lines", count, "keys"
