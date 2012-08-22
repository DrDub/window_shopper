from TrainGenerator import *;
from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;

import sys;
from nltk.stem.snowball import EnglishStemmer
import bsddb;

stemmer = EnglishStemmer();
model_factory = 0;
word_stat = 0;
scorer = CosTextScorer();
similarity_live = False
SIMILARITY_DIR = "/path/to/wnsimil"
SIMILARITY_URL = "http://127.0.0.1:10137/sim"

class WindowWorker:
    def __init__(self, window_chain):
        self.window_chain = window_chain;
    
    def work(self, text_piece):
        for window in text_piece.windows:
            self.window_chain.work(window);

class IDFExtractor:
    def __init__(self, word_stat):
        self.word_stat = word_stat;

    def extract(self, topic, doc, window):
        idfs = [];
        for token in window.tokens:
            if self.word_stat.has_key(token):
                idfs.append(np.log(self.word_stat['0']) - np.log(self.word_stat[token]));
        return np.mean(idfs), max(idfs), min(idfs);

class FidelityExtractor:
    def __init__(self, scorer):
        self.scorer = scorer;

    def extract(self, topic, doc, window):
       doc_window_score = self.scorer.score(doc.lm, window.lm);
       window_window_scores = [];
       for other_window in doc.windows:
            if other_window <> window:
                window_window_scores.append(self.scorer.score(window.lm, other_window.lm));
       return doc_window_score, np.mean(window_window_scores), max(window_window_scores), min(window_window_scores);

class RelevanceExtractor:
    def __init__(self, scorer):
        self.scorer = scorer;

    def extract(self,topic, doc, window):
        return scorer.score(topic.lm, window.lm), scorer.score(topic.lm, doc.lm);

class SimilarityExtractor:
    """
    Find which terms appear in the window similar to terms in the topic.
    """
    MAX = 1.
    MIN = -99999.
    def __init__(self, live,url=None,data_dir=None):
        self.live = live
        if live:
            self.url = url
        else:
            self.data_dir = data_dir
            self.simil_cache = dict()
            self.simil_error = set()

    def extract(self, topic, doc, window):
        window_token_set = set(window.lemmas)
        if len(window.lemmas) == 0:
            return SimilarityExtractor.MIN,
        found = [keyword for keyword in topic.lemmas if keyword in window_token_set]
        if len(found) > 1:
            # multiple keywords in window, max score
            return SimilarityExtractor.MAX,
        best_overall = SimilarityExtractor.MIN
        second_best = SimilarityExtractor.MIN
        for keyword in topic.lemmas:
            if keyword in found:
                continue
            best = max(map(lambda other: self.compute_similarity(keyword,other), window_token_set))
            if best >= best_overall:
                second_best = best_overall
                best_overall = best
            elif best > second_best:
                second_best = best
        if len(topic.lemmas) - len(found) == 1:
            return best_overall,
        else:
            if second_best == SimilarityExtractor.MIN:
                return SimilarityExtractor.MIN,
            else:
                return (best_overall + second_best) / 2,

    def compute_similarity(self, term1, term2):
        if self.live:
            from urllib import urlopen, quote_plus
            try:
                return float(urlopen("%s/%s/%s" % (self.url, quote_plus(term1), quote_plus(term2))).read())
            except Exception as e:
                print e
                return SimilarityExtractor.MIN
        else:
            if term1 in self.simil_error:
                return SimilarityExtractor.MIN                
            if not term1 in self.simil_cache:
                try:
                    f = open("%s/%s.scores" % (self.data_dir, term1))
                    scores = dict()
                    for line in f:
                        score, word = line.split()
                        scores[word] = float(score)
                    self.simil_cache[term1] = scores
                except Exception as e:
                    print e
                    self.simil_error.add(term1)
                    return SimilarityExtractor.MIN
            scores = self.simil_cache[term1]
            if not term2 in scores:
                return SimilarityExtractor.MIN
            else:
                return scores[term2]

def extract_window_feature(topic, doc, window):
    extractors = [ IDFExtractor(word_stat), FidelityExtractor(scorer), RelevanceExtractor(scorer),
        SimilarityExtractor(similarity_live, url=SIMILARITY_URL, data_dir=SIMILARITY_DIR) ];
    values = [];
    for extractor in extractors:
        values += extractor.extract(topic, doc, window);
    return values;

def exe_extract_feature(argv):
    window_path, doc_path, topic_path, judge_path, word_stat_path, out_path = argv;
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    window_db = bsddb.hashopen(window_path);
    doc_db = bsddb.hashopen(doc_path);
    global word_stat, model_factory;
    word_stat = load_word_stat(word_stat_path);
    model_factory = DocumentModelFactory(word_stat);
    writer = open(out_path, 'w');

    topic_chain = TextChain([TextStemmer(word_tokenize, stemmer), TextModeler(model_factory)]);
    window_chain = topic_chain;
    doc_chain = TextChain([TextStemmer(word_tokenize, stemmer), TextModeler(model_factory), WindowWorker(window_chain)])

    topic_ids = judge_file.keys()
    for topic_id in topic_ids:
        if not topics.has_key(topic_id):
            continue;
        topic_str = topics[topic_id];
        topic = TextPiece(topic_str);
        topic_chain.work(topic);

        for docno, rel in judge_file[topic_id].items()[:1]:
            if not is_cluewebB(docno) or not doc_db.has_key(docno) or not window_db.has_key(docno):
                continue;
            doc_str = doc_db[docno];
            windows_str = window_db[docno];
            doc = TextPiece(doc_str);
            doc.windows = map(lambda window_str: TextPiece(window_str), windows_str.split('\n'));
            doc_chain.work(doc);

            for i in xrange(len(doc.windows)):
                window = doc.windows[i];
                feature_values = extract_window_feature(topic, doc, window);
                window_id = '%s-%s-%d' % (topic_id, docno, i);
                feature_string = ','.join(map(str, feature_values));
                writer.write('%s:%s\n' % (window_id, feature_string));
    writer.close();


if __name__ == '__main__':
    exe_extract_feature(sys.argv[1:]);


