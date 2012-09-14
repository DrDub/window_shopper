from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
from TextUtil import *;
from WindowExtractor import *;

import numpy as np;
import time;
import sys;
import bsddb;
from nltk.stem.snowball import EnglishStemmer
from multiprocessing import Pool;
import traceback;

stemmer = EnglishStemmer();
stop_path = 'data/stoplist.dft';

'''
aggregate by average of k-nearest neighbors' distances
'''
class Aggregator:
    def __init__(self, K):
        self.K = K;

    def aggregate(self, values):
        values.sort(reverse = True);
        return np.mean(values[:self.K]);

class Document:
    def __init__(self, docno, doc, windows, rel):
        self.docno = docno;
        self.windows = windows;
        self.doc = doc;
        self.rel = rel;

class DocumentModelFactory:
    def __init__(self, word_stats):
        self.word_stats = word_stats;

    def build(self, tokens):
        token_tf = {};
        map(lambda token: token_tf.__setitem__(token, token_tf.get(token,0) + 1), tokens);
        doc_model = {};
        for token, tf in token_tf.items():
            idf = self.idf(token);
            if idf:
                doc_model[token] = tf * idf;
        return doc_model;

    def idf(self, token):
        if not self.word_stats.has_key(token):
            return 0;
        return np.log(self.word_stats['']/self.word_stats[token]);
            

class CosTextScorer:
    def __init__(self):
        self.max_score = 1.0;

    def score(self, model1, model2):
        return self.dot_product(model1, model2)/(self.norm(model1) * self.norm(model2))

    def dot_product(self, model1, model2):
        keys = set(model1.keys());
        keys = keys.intersection(model2.keys());
        val = 0;
        for key in keys:
            val += model1[key] * model2[key];
        return val;

    def norm(self, model):
        norm_val = np.sqrt(sum(map(lambda val: val ** 2, model.values())));
        return norm_val;


class RetrievalWindowRanker:
    def __init__(self, scorer, model_factory):
        self.scorer = scorer;
        self.topic_chain = TextChain([TextTokenizer(word_tokenize), TextTokenNormalizer(), TextStopRemover(stop_path), TextStemmer(stemmer), TextModeler(model_factory)]);
        self.window_chain = self.topic_chain;

    def rank(self, topic_str, docs):
        topic = TextPiece(topic_str);
        self.topic_chain.work(topic);        
        for doc in docs:
            doc.score_windows = [];
            for i in xrange(len(doc.windows)):
                window = doc.windows[i];
                self.window_chain.work(window);
                score = self.scorer.score(topic.lm, window.lm);
                if not doc.rel:
                    score = -score;
                doc.score_windows.append(([score], i));
            
class DistanceWindowRanker:
    def __init__(self, scorer, docmodel_factory, aggregators):
        self.scorer = scorer;
        self.window_chain = TextChain([TextTokenizer(word_tokenize), TextTokenNormalizer(), TextStopRemover(stop_path), TextStemmer(stemmer), TextModeler(docmodel_factory)]); 
        self.doc_chain = self.window_chain;
        self.aggregators = aggregators;

    def rank(self, query, docs):
        print 'doc num:', len(docs);
        sys.stdout.flush();
        #* build doc model
        for doc in docs:
            self.doc_chain.work(doc.doc);

        #* calculate the similarity
        for doc1 in docs:
            doc1.score_windows = [];
            for i in xrange(len(doc.windows)):
                window = doc.windows[i];
                self.window_chain.work(window);
                score = self.scorer.score(doc1.doc.lm, window.lm);
                other_scores = [];
                for doc2 in docs:
                    if (doc2.rel > 0 and doc1.rel <= 0) or (doc2.rel <= 0 and doc1.rel) > 0 :
                        other_scores.append(self.scorer.score(doc2.doc.lm, window.lm));
                values = map(lambda aggregator: score - aggregator.aggregate(other_scores), self.aggregators);
                doc1.score_windows.append((values, i));

judge_file = 0;
topics = 0;
word_stat = 0;
window_db = 0;
doc_db = 0;
ranker = 0;
K_options = [1,3,5,10];

def build_train(topic_id):
    try:
        print 'topic:', topic_id;
        docs = [];
        if not topics.has_key(topic_id):
            return docs;
        topic_str = topics[topic_id];
        for docno, rel in judge_file[topic_id].items():
            if not is_cluewebB(docno):
                continue;
            try:
                windows = map(lambda sentence: TextPiece(sentence), window_db[docno].split('\n'));
                docs.append(Document(docno, TextPiece(doc_db[docno]), windows, int(rel)));
            except Exception,e:
                sys.stderr.write(str(traceback.format_exc()));
                sys.stderr.write('%s %s: error at %s\n' % (str(e.__class__), str(e),docno));
                sys.stderr.write('-' * 100 + '\n');
                sys.exit(-1);
        ranker.rank(topic_str, docs);
    except Exception as e:
        print traceback.format_exc();
        sys.exit(-1);
    return docs;
                
def exe_build_train(argv):
#1. create the workers;
    judge_path, topic_path, word_stat_path, doc_path, window_path, out_path = argv;
    global judge_file, topics, doc_db, window_db, word_stat, ranker;
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    doc_db = bsddb.hashopen(doc_path);
    window_db = bsddb.hashopen(window_path);
    word_stat = load_word_stat(word_stat_path);
#    aggregators = map(lambda k: Aggregator(k), K_options);
#    ranker = DistanceWindowRanker(CosTextScorer(), DocumentModelFactory(word_stat),aggregators);
    ranker = RetrievalWindowRanker(CosTextScorer(), DocumentModelFactory(word_stat));

#2. build the training data;
#    p = Pool(4);
    topic_ids = judge_file.keys();
#    docs_groups = p.map(build_train, topic_ids);
    docs_groups = map(build_train, topic_ids);
    assert len(docs_groups) == len(topic_ids);

#3. write out the training data
    writer = open(out_path, 'w');
    for i in xrange(len(topic_ids)):
        topic_id = topic_ids[i];
        docs = docs_groups[i];
        for doc in docs:
            docno = doc.docno;
            judge = judge_file[topic_id][docno];
            for scores, sentence_id in doc.score_windows:
                score_str = ','.join(map(str, scores));
                writer.write('%s %s %s %d %s\n' % (topic_id, docno, judge, sentence_id, score_str));    
    writer.close();
        
def test_stat(argv):
    stat_path = argv[0];
    t0 = time.time();
    word_stat = WordStat(stat_path);
    print time.time() - t0;

def gen_field(writer, docno, sentence_lines, score_id, sentence_num):
    score_sentences = [];
    for sentence_line in sentence_lines:
        pos = sentence_line.find(':');
        scores = sentence_line[:pos];
        sentence = sentence_line[pos+1:];
        score = float(scores.split(',')[score_id]);
        score_sentences.append((score, sentence));
    score_sentences.sort();
    snippet = ' |||| '.join(map(lambda score_sentence: score_sentence[1], score_sentences[:sentence_num]));
    writer.write('>>>>docno:%s\n' % docno);
    writer.write('>>>>desc:%s\n' % snippet);
    writer.write('----END-OF-RECORD----\n');

def exe_gen_field(argv):
    snippet_path = argv[0];
    average_num = int(argv[1]);
    sentence_num = int(argv[2]);
    out_path = argv[3];
    score_id = K_options.index(average_num);

    f = open(snippet_path);
    writer = open(out_path, 'w');
    line = f.readline();
    docno = 0;
    sentence_lines = [];
    while line:
        line = line.strip();
        if line.startswith('docno:'):
            if docno:
                gen_field(writer, docno, sentence_lines, score_id, sentence_num);
                sentence_lines = [];
            docno = line[6:];
        elif line.startswith('topic_id:'):
            pass;
        else:
            sentence_lines.append(line);
        line = f.readline();
    writer.close();
    f.close();

def exe_view_train(argv):
    train_path, window_path, doc_path = argv;
    from Learner import TrainFile;
    train_file = TrainFile();
    train_file.load(train_path);
    window_db = bsddb.hashopen(window_path);
    doc_db = bsddb.hashopen(doc_path);

    num = 1000;
    key = train_file.keys()[num];
    qid, docno, rel, sid = key.split();
    doc_text = doc_db[docno];
    print qid, docno;
    print doc_text;
    print '=' * 50;
    windows = window_db[docno].split('\n'); 
    window_scores = [];
    for key in train_file.keys()[num:]:
        qid, curr_docno, rel, sid = key.split();
        if curr_docno <> docno:
            break;
        window_text = windows[int(sid)]; 
        value = train_file[key];
        window_scores.append((value, window_text));
    window_scores.sort();
    for score, window_text in window_scores:
        print score, window_text;

if __name__ == '__main__':
    option = sys.argv[1];
    if option == '--gen-train':
        exe_build_train(sys.argv[2:]);
    elif option == '--test-stat':
        test_stat(sys.argv[2:]);
    elif option == '--gen-field':
        exe_gen_field(sys.argv[2:]);
    elif option == '--view-train':
        exe_view_train(sys.argv[2:]);
        
