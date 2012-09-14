import bsddb;
import sys;

in_path, out_path = sys.argv[1:]
db = bsddb.hashopen(in_path);
f = open(out_path, 'w');
i = 0;
for key, value in db.iteritems():
    f.write('%s\n%s\n' % (key, value));
    f.write('+=' * 50 + '\n');
    f.flush();
f.close();
db.close();
