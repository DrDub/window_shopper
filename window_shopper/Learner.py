import bsddb;

class TrainFile(dict):
    def keys(self):
        sorted_keys = dict.keys(self);
        sorted_keys.sort();
        return sorted_keys;

    def load(self, path):
        f = open(path);
        line = f.readline();
        while line:
            tokens = line.strip().split();
            key = ' '.join(tokens[:4]);
            value = ' '.join(tokens[4:]);
            self[key] = value;
            line = f.readline();
        f.close();

    def store(self, path):
        f = open(path, 'w');
        keys = self.keys();
        for key in keys:
            f.write('%s %s\n' % (key, self[key]));
        f.close();

def exe_merge(target_path, feature_path, out_path):
    target_file = TrainFile();
    target_file.load(target_path);
    feature_file = TrainFile();
    feature_file.load(feature_path);
    train_file = TrainFile();

    keys = target_file.keys();
    for key in keys:
        if feature_file.has_key(key):
            value = '%s,%s'  % (target_file[key], feature_file[key]); 
            train_file[key] = value;
    train_file.store(out_path);

'''
training file -> svm-rank file 
'''
def exe_train_to_svmrank(argv):
    train_path, svm_path = argv;
    train_file = TrainFile();
    train_file.load(train_path);
    svm_writer = open(svm_path, 'w');

    qid = 0;
    prev_docno = 0;
    keys = train_file.keys();
    for key in keys:
        topic_id, docno, rel, sid = key.split();
        value_tokens = train_file[key].split();
        if len(value_tokens) == 2:
            score, feature_str = value_tokens;
            features = feature_str.split(',');
            if not prev_docno or docno <> prev_docno:
                qid += 1;
                prev_docno = docno;
            svm_line = '%s qid:%d' % (score, qid);
            for i in xrange(len(features)):
                svm_line += ' %d:%s' % (i+1, features[i]);
            svm_line += '#%s' % key;
            svm_writer.write(svm_line + '\n');
    svm_writer.close();

def exe_svmrank_to_target(argv):
    svm_path, predict_path, target_path = argv;
    svm_file = open(svm_path);
    predict_file = open(predict_path);
    target_file = TrainFile();

    predict_line = predict_file.readline();
    svm_line = svm_file.readline();
    while predict_line and svm_line:
        score = float(predict_line.strip());
        svm_content, svm_comment = svm_line.strip().split('#');
        target_file[svm_comment] = str(score);
        
        predict_line = predict_file.readline();
        svm_line = svm_file.readline();
    target_file.store(target_path);

def gen_snippet(window_db, docno, scores, writer, sen_num):
    windows = window_db[docno].split('\n');
    scores.sort(reverse=True);
    if len(windows) == 1:
        snippet = windows[0];
    else:
        snippet = ' |||| '.join(map(lambda score_sid: windows[score_sid[1]], scores[:sen_num]));
    writer.write('>>>>docno:%s\n' % docno);
    writer.write('>>>>desc:%s\n' % snippet);
    writer.write('----END-OF-RECORD----\n');

def exe_target_to_snippet(argv):
    target_path, window_path, snippet_path, sen_num = argv;
    sen_num = int(sen_num);

    target_file = TrainFile();
    target_file.load(target_path);
    window_db = bsddb.hashopen(window_path);
    snippet_writer = open(snippet_path, 'w');

    prev_docno = 0;
    scores = [];
    for key in target_file.keys():
        qid, docno, rel, sid = key.split();
        score = target_file[key];
        scores.append((float(score), int(sid)));
        if not prev_docno:
            prev_docno = docno;
        elif docno <> prev_docno:
            gen_snippet(window_db, prev_docno, scores, snippet_writer, sen_num);
            prev_docno = docno;
            scores = [];
    gen_snippet(window_db, prev_docno, scores, snippet_writer, sen_num);
    snippet_writer.close();
    window_db.close();


if __name__ == '__main__':
    import sys;
    argv = sys.argv;
    option = argv[1];
    
    if option == '--merge-train':
        exe_merge(* argv[2:]);
    elif option == '--train2svm':
        exe_train_to_svmrank(argv[2:]);
    elif option == '--svm2target':
        exe_svmrank_to_target(argv[2:]);
    elif option == '--gen-snippet':
        exe_target_to_snippet(argv[2:]);
    else:
        print 'error param!';
