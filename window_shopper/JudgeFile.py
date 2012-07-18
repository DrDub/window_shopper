import util;
import os;

class PRelFile(util.KeyMapMap):
    def __init__(self, path):
        util.KeyMapMap.__init__(self, 0);
        self._load(path);

    def _load(self, path):
        f = open(path);
        lines = map(str.strip, f.readlines());
        for line in lines:
            tokens = line.split();
            qid, docid, score = tokens[:3];
            qid = int(qid);
            self.add(qid, docid, score);


class QRelFile(util.KeyMapMap):
    def __init__(self, path):
        util.KeyMapMap.__init__(self, 0);
        self._load(path);

    def _load(self, path):
        f = open(path);
        lines = map(str.strip, f.readlines());
        for line in lines:
            tokens = line.split();
            qid, nothing, docid, score = tokens[:4];
            qid = int(qid);
            if docid == '1':
                print line;
            self.add(qid, docid, score);

    def store(self, path):
        f = open(path, 'w');
        for qid in self.keys():
            for docid, score in self._data[qid].items():
                f.write('%d 0 %s %s\n' % (qid, docid, score));
        f.close();

def main():
    pass;

if __name__ == '__main__':
    main();

