from util import KeyValueMap;

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

        topics = {};
        query_id = 0;
        query_str = '';
        for line in lines:
            if line.startswith('<DOC'):
                query_id = int(line.split()[1][:-1]);
            elif line.startswith('</DOC>'):
                topics[query_id] = query_str;
                query_str = '';
            else:
                query_str += line + ' ';
        return topics;

    def write(self, topics, path):
        f = open(path, 'w');
        ids = topics.keys();
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

def exe_convert(informat, inpath, outformat, outpath):
    informat = get_format(informat);
    outformat = get_format(outformat);
    topics = informat.read(inpath);
    outformat.write(topics, outpath);

def exe_intersection(topic1_path, topic2_path, out_topic_path):
    '''
        generate a topic file containing the same content of topic file 1, but only containing the topic ids in topic file 2;
    '''
    io = StandardFormat();
    topics1 = io.read(topic1_path);
    topics2 = io.read(topic2_path);
    remove_keys = set(topics1.keys());
    remove_keys.difference_update(topics2.keys());
    map(lambda topic_id: topics1.__delitem__(topic_id), remove_keys);
    io.write(topics1, out_topic_path);

if __name__ == '__main__':
    import sys;
    option = sys.argv[1];
    argv = sys.argv[2:];
    if option == '--convert':
        exe_convert(*argv);
    elif option == '--intersection':
        exe_intersection(*argv);
    else:
        print 'error param!';
