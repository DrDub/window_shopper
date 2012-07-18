import heapq;
import sys;

'''
a heap that can keep a number of maximum values
'''
class Heap:
    def __init__(self, dat=None, key=lambda x:x, size=sys.maxint):
        self.size = size;
        if dat:
            self.dat = [(key(item), item) for item in dat];
            heapq.heapify(self.dat);
        else:
            self.dat = [];

    def push(self, item):
        if len(self.dat) > self.size:
            heapq.heappushpop(self.dat, item);
        else:
            heapq.heappush(self.dat, item);

    def pop(self):#pop the smallest item
        key, item = heapq.pop(self.dat);
        return item;
