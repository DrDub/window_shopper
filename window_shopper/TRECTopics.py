from util import KeyValueMap;

class Topics(KeyValueMap):
    def __init__(self):
        KeyValueMap.__init__(self, '');

def get_format(name):
    if name == 'standard':
        return StandardFormat();
    elif name == 'flat':
        return FlatFormat();
    raise 'I cannot recognize the format name:' + name;

class StandardFormat:
    def __str__(self):
        return 'standard';

    def read(self, path):
        f = open(path);
        lines = map(str.strip, f.readlines());
        f.close();

        topics = Topics();
        query_id = 0;
        query_str = '';
        for line in lines:
            if line.startswith('<DOC'):
                query_id = int(line.split()[1][:-1]);
            elif line.startswith('</DOC>'):
                topics.add(query_id, query_str);
                query_str = '';
            else:
                query_str += line + ' ';

        return topics;

    def write(self, topics, path):
        f = open(path, 'w');
        ids = topics.getIDs();
        for query_id in ids:
            query_str = topics.get(query_id);
            f.write('<DOC %d>\n%s\n</DOC>\n' % (query_id, '\n'.join(query_str.split())));
        f.close();
        

class FlatFormat:
    def __str__(self):
        return 'flat';

    def read(self, path):
        topics = Topics();
        f = open(path);
        for line in f.readlines():
            query_id, query_str = line.strip().split(':');
            query_id = int(query_id);
            topics.add(query_id, query_str);
        f.close();
        return topics;
    
    def write(self, topics, path):
        f = open(path, 'w');
        ids = topics.getIDs();
        for query_id in ids:
            f.write('%d:%s\n' % (query_id, topics.get(query_id)));
        f.close();

def convert(informat, inpath, outformat, outpath):
    topics = informat.read(inpath);
    outformat.write(topics, outpath);

if __name__ == '__main__':
    import sys;
    informat = get_format(sys.argv[1]);
    inpath = sys.argv[2];
    outformat = get_format(sys.argv[3]);
    outpath = sys.argv[4];
    print 'converting %s(%s) to %s(%s)' % (inpath, informat, outpath, outformat);
    convert(informat, inpath, outformat, outpath);
