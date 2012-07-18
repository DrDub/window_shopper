
class KeyValueMap(dict):
    def __init__(self, default_value):
        self._default_value = default_value;

    def add(self, key, value):
        self[key] = value;

    def keys(self):
        keys = dict.keys(self);
        keys.sort();
        return keys;

    def values(self):
        values = self.values();
        values.sort();
        return values;

    def get(self, key):
        if self.has_key(key):
            return self[key];
        else:
            return self._default_value;

    def set(self, key, value):
        self[key] = value;

    def merge(self, another_map):
        other_keys = another_map.keys();
        for key in other_keys:
            if self.has_key(key):
                raise Exception("key %s confilict in merging!" % str(key));
            self.set(key, another_map.get(key));

    def filter_keys(self, valid_key_set):#only keep the keys in the valid_key_set;
        for key in self.keys():
            if not valid_key_set.__contains__(key):
                self.__delitem__(key);

class KeyArrayMap(KeyValueMap):
    def __init__(self):
       KeyValueMap.__init__(self, []); 

    def add(self, key, value):
        if not self.has_key(key):
            self[key] = [];
        self[key].append(value);

    def values(self):
        results = reduce(lambda part_values, key: part_values.union(self[key]), self.keys(), set());
        results = list(results);
        results.sort();
        return results;

class KeyMapMap(KeyValueMap):
    def __init__(self, default_value):
        KeyValueMap.__init__(self, {});
        self._default_real_value = default_value;

    def add(self, key1, key2, value):
        if not self.has_key(key1):
            self[key1] = {};
        self[key1][key2] = value;

    def get_value(self, key1, key2):
        if self.has_key(key1):
            son_map = self[key1];
            if son_map.has_key(key2):
                return son_map[key2];
        return self._default_real_value;

    def key2s(self):
        results = reduce(lambda part_values, key1: part_values.union(self[key1].keys()), self.keys(), set());
        results = list(results);
        results.sort();
        return results;

    def values(self):
        results = reduce(lambda part_values, key1: part_values.union(self[key1].values()), self.keys(), set());
        results = list(results);
        results.sort();
        return results;

    def filter_key2s(self, valid_key_set):
        for key, submap in self.items():
            for key2 in submap.keys():
                if not valid_key_set.__contains__(key2):
                    submap.__delitem__(key2);

def test():
    map3 = KeyMapMap(0);
    map3.add(1,2,3);
    map3.add(1,4,5);
    map3.add(2,2,4);
    print map3.get(1,2), map3.get(2,2), map3.get(3,1), map3.get(1,3);
    print map3.keys();
    print map3.key2s();
    print map3.values();

if __name__ == '__main__':
    test();
