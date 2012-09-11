import ContentRecord;
import re;

def load(path):
    reader = ContentRecord.ContentReader(path);
    record_dict = {};
    record = reader.next();
    while record:
        topic_id = record.get('query-ID');
        if not record_dict.has_key(topic_id):
            record_dict[topic_id] = [];
        record_dict[topic_id].append(record);
        record = reader.next();
    return record_dict;

def select(record_dict, n):
    import random;
    selected_dict = {};
    for topic_id in record_dict.keys():
        records = record_dict[topic_id]
        sample_records = random.sample(records, min(n, len(records)));
        selected_dict[topic_id] = sample_records;
    return selected_dict;

def write(record_dict, path):
    f = open(path, 'w');
    topic_ids = record_dict.keys();
    topic_ids.sort();
    for topic_id in topic_ids:
        for record in record_dict[topic_id]:
            f.write(record.__str__() + '\n');
    f.close();

def test_load(path):
    path = argv[0];
    record_dict = load(path);
    print len(record_dict.keys());
    print len(filter(lambda records: len(records) >= 3, record_dict.values()));

def exe_select(in_path, n, out_path):
    '''
        select a subset of docs for manual judgement randomly;
    '''
    n = int(n);
    record_dict = load(in_path);
    record_dict = select(record_dict, n);
    write(record_dict, out_path);

def parse_feature_line(feature_line):
    topic_id, docno, rel, sentence_id, features = feature_line.strip().split();
    features = features.split(',');
    return topic_id, docno, rel, sentence_id, features;

def insert_pred_dict(pred_dict, topic_id, docno, sentence_id, score):
    if not pred_dict.has_key(topic_id):
        pred_dict[topic_id] = {};
    if not pred_dict[topic_id].has_key(docno):
        pred_dict[topic_id][docno] = [];
    pred_dict[topic_id][docno].append((score, sentence_id));

def load_pred(pred_path, feature_path, feature_num):
    pred_dict = {};

    pred_reader = open(pred_path);
    feature_reader = open(feature_path);
    pred_line = pred_reader.readline();
    feature_line = feature_reader.readline();
    while pred_line and feature_line:
        topic_id, docno, rel, sentence_id, features = parse_feature_line(feature_line);
        while len(features) <> feature_num:
            insert_pred_dict(pred_dict, topic_id, docno, sentence_id, 0)            
            feature_line = feature_reader.readline();
            topic_id, docno, rel, sentence_id, features = parse_feature_line(feature_line);
        score = float(pred_line.strip());
        insert_pred_dict(pred_dict, topic_id, docno, sentence_id, score)            
        pred_line = pred_reader.readline();
        feature_line = feature_reader.readline();
    pred_reader.close();
    feature_reader.close();

    return pred_dict;

def exe_merge_pred(pred_path, feature_path, feature_num, out_path):
    '''
        merge prediction file and feature file, and get sentence ranking for each (topic, doc) according to the predicted score;
    '''
    feature_num = int(feature_num);
    pred_dict = load_pred(pred_path, feature_path, feature_num);
    writer = open(out_path, 'w');
    topic_ids = pred_dict.keys();
    topic_ids.sort();
    for topic_id in topic_ids:
        doc_info = pred_dict[topic_id];
        docnos = doc_info.keys();
        docnos.sort();
        for docno in docnos:
            sentence_scores = doc_info[docno];
            sentence_scores.sort(reverse=True);
            sentence_ids = map(lambda sentence_score: sentence_score[1], sentence_scores);
            writer.write('%s %s %s\n' % (topic_id, docno, ','.join(sentence_ids)));
    writer.close();

def exe_align_snippet(senrank_path, window_path, pilot_snippet_path, out_path):
    '''
        generate the snippet file from the sentence ranking file;
        only generate the snippets in the pilot snippet path;
    '''
    senrank_dict = {};
    senrank_file = open(senrank_path);
    line = senrank_file.readline();
    while line:
        topic_id, docno, senrank = line.strip().split();
        senrank = map(int, senrank.split(','));
        senrank_dict[docno] = senrank;
        line = senrank_file.readline();
    senrank_file.close();

    import bsddb;
    window_db = bsddb.hashopen(window_path);
    pilot_reader = ContentRecord.ContentReader(pilot_snippet_path);
    record = pilot_reader.next();
    writer = open(out_path, 'w');
    total, found = 0, 0;
    while record:
        total += 1;
        docno = record.get('ex-ID').strip();
        desc = record.get('desc').strip();
        new_record = ContentRecord.ContentRecord(record.data);
        new_desc = '';
        if senrank_dict.has_key(docno):
            sentences = window_db[docno].split('\n');
            sentence_ids = senrank_dict[docno];
            for sentence_id in sentence_ids:
                new_desc += sentences[sentence_id].strip() + ' ';
                if len(new_desc) > len(desc):
                    break;
            found += 1;
        else:
            sys.stderr.write('cannot find the doc %s\n' % docno);
        new_desc = new_desc.strip();
        new_desc, num = re.subn('[\\t|\s]+', ' ', new_desc);
        new_record.set('desc', new_desc);
        writer.write(new_record.__str__() + '\n');
        record = pilot_reader.next();
    writer.close();
    print total, found;
        

if __name__ == '__main__':
    import sys;
    option = sys.argv[1];
    argv = sys.argv[2:];
    if option == '--test-load':
        test_load(*argv);
    elif option == '--select':
        exe_select(*argv);
    elif option == '--merge-pred':
        exe_merge_pred(*argv);
    elif option == '--align-snippet':
        exe_align_snippet(*argv);
