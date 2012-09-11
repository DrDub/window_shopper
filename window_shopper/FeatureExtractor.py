from TrainGenerator import *;
from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;



task_num = 4;

stemmer = EnglishStemmer();
model_factory = 0;
word_stat = 0;
window_db = 0;
doc_db = 0;
scorer = CosTextScorer();

topic_chain = 0;
doc_chain = 0;
window_chain = 0;
topic = 0;
topic_id = 0;


class WindowWorker:
    def __init__(self, window_chain):
        self.window_chain = window_chain;
    
    def work(self, text_piece):
        for window in text_piece.windows:
            self.window_chain.work(window);

class DocumentTitleWorker:
    def __init__(self, title_chain):
        self.title_chain = title_chain;

    def work(self, text_piece):
        text_piece.title = TextPiece(text_piece.text[:text_piece.text.find('\n')].encode('utf8'));
        self.title_chain.work(text_piece.title);

class ExtractorError(Exception):
    def __init__(self, msg):
        self.msg = msg;

    def __str__(self):
        return self.msg;

'''
require:
    1) tokenized;
    2) stop words removed;
    3) stemmed;
    4) sentence tokenized;
result:
    text_piece.significant_terms;
'''
class SignificantTermDetector:
    def work(self, text_piece):
        sentence_num = len(text_piece.sentences);
        term_counts = {};
        for token in text_piece.tokens:
            term_counts.__setitem__(token, term_counts.get(token, 0) + 1);
        term_count_list = term_counts.items(); 
        term_count_list.sort(reverse=True, key=lambda term_count: term_count[1]);

        term_num = 7;
        if sentence_num > 40:
            term_num += .1 * (sentence_num - 40);
        elif sentence_num < 25:
            term_num -= .1 * (25 - sentence_num);
        text_piece.significant_terms = map(lambda term_count: term_count[0], term_count_list[:term_num]);

class QueryFeatureExtractor:
    def __init__(self, word_stat):
        self.word_stat = word_stat;

    def extract(self, topic, doc, window):
        topic_term_num = len(topic.tokens);
        idfs = [];
        for token in topic.tokens:
            if self.word_stat.has_key(token):
                idfs.append(np.log(self.word_stat['0']) - np.log(self.word_stat[token]));
        return topic_term_num, np.mean(idfs);    

class IDFExtractor:
    def __init__(self, word_stat):
        self.word_stat = word_stat;

    def extract(self, topic, doc, window):
        idfs = [];
        for token in window.tokens:
            if self.word_stat.has_key(token):
                idfs.append(np.log(self.word_stat['']) - np.log(self.word_stat[token]));
        if len(idfs) == 0:
            raise ExtractorError('no valid token in the window "%s"' % window.text.encode('utf8'));
        return np.mean(idfs), max(idfs), min(idfs);

class FidelityExtractor:
    def __init__(self, scorer):
        self.scorer = scorer;

    def extract(self, topic, doc, window):
        doc_window_score = self.scorer.score(doc.lm, window.lm);
        window_window_scores = [];
        is_title = 0;
        if doc.title.text.find(window.text) >= 0:
            is_title = 1;
            #print 'title:',doc.title.text
            #print 'text:', window.text
        title_window_score = self.scorer.score(doc.title.lm, window.lm);
        if len(doc.windows) == 1:
            return doc_window_score, self.scorer.max_score, self.scorer.max_score;
        for other_window in doc.windows:
            if other_window <> window:
                window_window_scores.append(self.scorer.score(window.lm, other_window.lm));
        return is_title, title_window_score, doc_window_score, np.mean(window_window_scores), max(window_window_scores), min(window_window_scores);

class RelevanceExtractor:
    def __init__(self, scorer):
        self.scorer = scorer;

    def extract(self,topic, doc, window):
        topic_term_set = set(topic.tokens);
        matched_terms = filter(lambda token: topic_term_set.__contains__(token), window.tokens);
        matched_term_set = set(matched_terms);
        return scorer.score(topic.lm, window.lm), scorer.score(topic.lm, doc.lm);
        #return len(matched_terms)/float(len(window.tokens)), len(matched_term_set)/len(topic.tokens), len(matched_terms), len(matched_term_set), scorer.score(topic.lm, window.lm), scorer.score(topic.lm, doc.lm);

def extract_window_feature(topic, doc, window):
    extractors = [RelevanceExtractor(scorer)];
    #extractors = [IDFExtractor(word_stat), FidelityExtractor(scorer), RelevanceExtractor(scorer)];
    values = [];
    value_num = 0;
    try:
        for extractor in extractors:
            values += extractor.extract(topic, doc, window);
    except ExtractorError as e:
        sys.stderr.write(e.__str__() + '\n');
        sys.stderr.flush();
        values = [];
    return values;

def multithread_extract_feature(docno_rel_pair):
    lines = [];
    try:
        docno, rel = docno_rel_pair;
        if not is_cluewebB(docno) or not doc_db.has_key(docno) or not window_db.has_key(docno):
            return lines;
        doc = 0;
        doc_str = doc_db[docno];
        windows_str = window_db[docno];
        #print doc_str;
        #print '=' * 100;
        #print windows_str;
        if not doc_str or not windows_str:
            return lines;
        doc = TextPiece(doc_str);
        doc.windows = map(lambda window_str: TextPiece(window_str), windows_str.split('\n'));
        doc_chain.work(doc);

        for i in xrange(len(doc.windows)):
            window = doc.windows[i];
            feature_values = extract_window_feature(topic, doc, window);
            feature_string = ','.join(map(str, feature_values));
            line = '%s %s %s %d %s' % (topic_id, docno, rel, i, feature_string);
            lines.append(line);
        #print '\n'.join(lines);
        #sys.exit(-1);
    except Exception, e:
        print traceback.format_exc();
        sys.exit(-1);
    return lines;

def exe_extract_feature(argv):
    window_path, doc_path, topic_path, judge_path, word_stat_path, out_path = argv; 
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    global window_db, doc_db, word_stat, model_factory;
    window_db = bsddb.hashopen(window_path);
    doc_db = bsddb.hashopen(doc_path);
    word_stat = load_word_stat(word_stat_path);
    model_factory = DocumentModelFactory(word_stat);
    writer = open(out_path, 'w');

    global topic_chain, window_chain, doc_chain;
    topic_chain = TextChain([TextTokenizer(word_tokenize), TextStopRemover(stop_path), TextStemmer(stemmer), TextModeler(model_factory)]);
    window_chain = topic_chain;
    doc_chain = TextChain([TextTokenizer(word_tokenize),TextStopRemover(stop_path),  TextStemmer(stemmer), TextModeler(model_factory), WindowWorker(window_chain), DocumentTitleWorker(topic_chain)])

    global topic_id;
    topic_ids = judge_file.keys();
    for topic_id in topic_ids:
        if not topics.has_key(topic_id):
            continue;
        topic_str = topics[topic_id];
        print topic_id;
        global topic;
        topic = TextPiece(topic_str);
        topic_chain.work(topic);

        p = Pool(task_num);
        lines_group = p.map(multithread_extract_feature, judge_file[topic_id].items());
        for lines in lines_group:
            for line in lines:
                writer.write(line);
                writer.write('\n');
    writer.close();
    
#def exe_merge_feature(feature_path1, feature_path2, out_path):
    


if __name__ == '__main__':
    exe_extract_feature(sys.argv[1:]);
