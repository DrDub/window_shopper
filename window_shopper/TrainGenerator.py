from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;
from WindowExtractor import *;
from Heap import Heap;

import numpy as np;
import time;
import sys;
import bsddb;
from nltk.stem.snowball import EnglishStemmer
from multiprocessing import Pool;

stemmer = EnglishStemmer();

class TextChain:
    def __init__(self, workers):
        self.workers = workers;

    def work(self, text_piece):
        for worker in self.workers:
            worker.work(text_piece);

class TextStemmer:
    def __init__(self, tokenize_func, stemmer):
        self.tokenize_func = tokenize_func;
        self.stemmer = stemmer;

    def work(self, text_piece):
        text_piece.tokens = map(lambda token: self.stemmer.stem(token.lower()), self.tokenize_func(text_piece.text));

class TextModeler:
    def __init__(self, model_factory):
        self.model_factory = model_factory;

    def work(self, text_piece):
        text_piece.lm = self.model_factory.build(text_piece.tokens);

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
    def __init__(self, docno, windows, rel):
        self.docno = docno;
        self.windows = windows;
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
        self.topic_chain = TextChain([TextStemmer(word_tokenize, stemmer), TextModeler(model_factory)]);
        self.window_chain = self.topic_chain;

    def rank(self, topic_str, docs):
        topic = TextPiece(topic_str);
        self.topic_chain.work(topic);        
        for doc in docs:
            doc.score_windows = [];
            for window in doc.windows:
                self.window_chain.work(window);
                score = self.scorer.score(topic.lm, window.lm);
                if not doc.rel:
                    score = -score;
                doc.score_windows.append(([score], window.text));
            

class DistanceWindowRanker:
    def __init__(self, scorer, docmodel_factory, aggregators):
        self.scorer = scorer;
        self.window_chain = TextChain([TextStemmer(word_tokenize, stemmer), TextModeler(model_factory)]); 
        self.aggregators = aggregators;

    def rank(self, query, docs):
        print query, len(docs);
        sys.stdout.flush();
        #* build doc model
        print 'building model......';
        for doc in docs:
            for window in doc.windows:
                self.window_chain.work(window);

        #* partition windows into positive and negative;
        print 'partitioning windows......';
        positive_windows = [];
        negative_windows = [];
        for doc in docs:
            if doc.rel:
                positive_windows += doc.windows;
            else:
                negative_windows += doc.windows;
        print len(positive_windows), len(negative_windows);

        #* calculate the similarity
        print 'calculating similarity......';
        posneg_data = {};
        negpos_data = {};
        for window1 in positive_windows:
            id1 = id(window1);
            for window2 in negative_windows:
                id2 = id(window2);
                score = self.scorer.score(window1.lm, window2.lm);
                window1_scores = posneg_data.get(id1, Heap(size=10));
                window1_scores.push(score);
                posneg_data[id1] = window1_scores;
                window2_scores = negpos_data.get(id2, Heap(size=10));
                window2_scores.push(score);
                negpos_data[id2] = window2_scores;
        #print negpos_data.keys();
        #print posneg_data.keys();

        #* aggregate and rank the windows
        for doc in docs:
            score_windows = [];
            if doc.rel:
                score_data = posneg_data; 
            else:
                score_data = negpos_data;
            for window in doc.windows:
                raw_scores = score_data[id(window)].dat; 
                scores = map(lambda aggregator: aggregator.aggregate(raw_scores), self.aggregators);
                score_windows.append((scores, window.text));
            doc.score_windows = score_windows;

judge_file = 0;
topics = 0;
word_stat = 0;
window_db = 0;
ranker = 0;
K_options = [1,3,5,10];

def build_train(topic_id):
    print topic_id;
    docs = [];
    if not topics.has_key(topic_id):
        return docs;
    topic_str = topics[topic_id];
    for docno, rel in judge_file[topic_id].items():
        if not is_cluewebB(docno):
            continue;
        try:
            windows = map(lambda sentence: TextPiece(sentence), window_db[docno].split('\n'));
            docs.append(Document(docno, windows, int(rel)));
        except Exception,e:
            sys.stderr.write('%s: error at %s\n' % (str(e),docno));
    ranker.rank(topic_str, docs);
    return docs;
                
def exe_build_train(argv):
    judge_path, topic_path, word_stat_path, window_path, out_path = argv;
    global judge_file, topics, window_db, word_stat, ranker;
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    window_db = bsddb.hashopen(window_path);
    word_stat = load_word_stat(word_stat_path);
    #aggregators = map(lambda k: Aggregator(k), K_options);
    #ranker = DistanceWindowRanker(CosTextScorer(), DocumentModelFactory(word_stat),aggregators);
    ranker = RetrievalWindowRanker(CosTextScorer(), DocumentModelFactory(word_stat));
    print 'loaded......';

    p = Pool(8);
    topic_ids = judge_file.keys();
    docs_groups = p.map(build_train, topic_ids);
    assert len(docs_groups) == len(topic_ids);

    writer = open(out_path, 'w');
    for i in xrange(len(topic_ids)):
        topic_id = topic_ids[i];
        docs = docs_groups[i];
        writer.write('topic_id:%s\n' % (topic_id));
        for doc in docs:
            writer.write('docno:%s\n' % doc.docno);
            for scores, window_text in doc.score_windows:
                writer.write('%s:%s\n' % (','.join(map(str, scores)), window_text.encode('utf8')));
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


if __name__ == '__main__':
    option = sys.argv[1];
    if option == '--gen-train':
        exe_build_train(sys.argv[2:]);
    elif option == '--test-stat':
        test_stat(sys.argv[2:]);
    elif option == '--gen-field':
        exe_gen_field(sys.argv[2:]);
