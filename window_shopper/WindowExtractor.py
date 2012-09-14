from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;
from TextUtil import *;

import bsddb;
import os;
import subprocess;
import sys;
from multiprocessing import Pool;
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize;
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np;


suffixes = ['html', 'text', 'title'];
extract_script = 'extract-chunks-from-page.py';

def extract_text(docno, index_path):
    text = '';
    try:
        index = Index(index_path);
        content = index.get_doc_content(docno);
        html_path, text_path, title_path = map(lambda suffix: '%s.%s' % (docno, suffix), suffixes);
        f = open(html_path, 'w');
        f.write(content);
        f.close();

        subprocess.call(['python', extract_script, html_path, text_path, title_path]);

        title_f = open(title_path);
        # first line is the title
        text = ' '.join(map(str.strip, title_f.readlines())) + '\n';
        text_f = open(text_path);
        text += ''.join(text_f.readlines());
        
        #os.remove(html_path);
        #os.remove(text_path);
        #os.remove(title_path);
    except Exception, e:
        sys.stderr.write('error at docno %s\n' % docno);
    return text;

def is_cluewebB(docno):
    col_name = docno.split('-')[1];
    no = int(col_name[4:]);
    if col_name.startswith('en00') and no <= 11:
        return True;
    if col_name.startswith('enwp') and no <= 3:
        return True;
    return False;

def exe_extract_text(judge_path, index_path, text_db_path):
    judge_file = QRelFile(judge_path);
    docnos = judge_file.key2s();
    docnos = filter(is_cluewebB, docnos);
    #docnos = docnos[:1000];
    print 'doc number:', len(docnos);
    db = bsddb.hashopen(text_db_path, 'w');
    count = 0;
    texts = fastmap.fastmap(lambda docno: extract_text(docno, index_path), 30, docnos);
    assert len(docnos) == len(texts);
    for i in xrange(len(docnos)): 
        db[docnos[i]] = texts[i];
    db.close();

class TextPiece:
    def __init__(self, text):
        self.text = text.decode('utf8');
        self.tokenize();

    '''
    parse a piece of text to a vector of stemmed words
    '''    
    def tokenize(self):
        terms = word_tokenize(self.text);
        self.tokens = [];
        self.lemmas = []
        stemmer = EnglishStemmer();
        lemmatizer = WordNetLemmatizer()
        for term in terms:
            try:
                self.tokens.append(stemmer.stem(term).lower())
                self.lemmas.append(lemmatizer.lemmatize(term.lower()))
            except Exception, e:
                print 'current text:', self.text;
                print 'current term:', term;
                print str(e);
                sys.exit(-1);

def test_extract_text(judge_path, index_path):
    judge_file = QRelFile(judge_path);
    docnos = judge_file.key2s();
    print 'doc number:', len(docnos);
    for docno in filter(is_cluewebB, docnos)[:3]:
        text = extract_text(docno, index_path);
        print text
        print '-' * 20



def match_window(topic, doc_text, sentence_chain):
    topic_terms = set(topic.tokens);
    lines = doc_text.split('\n');
    candidates = [];
    for line in lines:
        sentences = sent_tokenize(line);
        is_candidate = False;
        for sentence_str in sentences:
            sentence = TextPiece(sentence_str);
            sentence_chain.work(sentence);
            if len(topic_terms.intersection(sentence.tokens)) > 0:
                candidates.append(sentence);
    return candidates;

def exe_extract_windows(argv):
    topic_path, judge_path, text_db_path, windows_db_path = argv;
    text_db = bsddb.hashopen(text_db_path);
    window_db = bsddb.hashopen(windows_db_path, 'w');
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    topic_chain = TextChain([TextTokenizer(word_tokenize), TextStopRemover('data/stoplist.dft'), TextStemmer(EnglishStemmer()), TextTokenNormalizer()]); 
    sentence_chain = TextChain([TextTokenizer(word_tokenize), TextStemmer(EnglishStemmer()), TextTokenNormalizer()]);
    for topic_id, topic_str in topics.items():
        print topic_id;
        sys.stdout.flush();
        topic = TextPiece(topic_str);
        topic_chain.work(topic);
        if not judge_file.has_key(topic_id):
            continue;
        docnos = judge_file[topic_id].keys();
        for docno in docnos:
            if not is_cluewebB(docno):
                continue;
            doc_text = text_db[docno];
            window_candidates = match_window(topic, doc_text, sentence_chain);
            sentences = map(lambda text_piece: text_piece.text, window_candidates);
            text = '\n'.join(sentences);
            window_db[docno] = text.encode('utf8');
    window_db.close();

def exe_extract_words(argv):
    windows_db_path, word_list_path = argv;
    window_db = bsddb.hashopen(windows_db_path);
    words = set();
    p = Pool(20);
    words_vec = p.map(word_tokenize, window_db.values());
    words = [];
    for part_words in words_vec:
        words += part_words;

    print 'writing.....'
    word_list_file = open(word_list_path, 'w');
    words = map(lambda word: word.lower(), words);
    words.sort();
    map(lambda word:word_list_file.write('%s\n' % word), words);
    word_list_file.close();

def exe_extract_topic_words(argv):
    from nltk.tokenize import word_tokenize;
    topic_path, word_list_path = argv;
    trec_format = StandardFormat();
    word_list_file = open(word_list_path, 'w');    
    topics = trec_format.read(topic_path);
    word_set = set();
    for topic_id, topic_text in topics.items():
        words = map(lambda word: word.lower(), word_tokenize(topic_text));
        word_set.update(words);
    word_list_file.write('\n'.join(word_set));
    word_list_file.close();

def exe_gen_word_db(argv):
    word_stat_path, word_db_path = argv;
    word_db = bsddb.hashopen(word_db_path, 'w');
    lines = open(word_stat_path).readlines();
    for line in lines:
        word, cf, df = line.strip().split();
        word_db[word] = '%s %s' % (cf, df);
    word_db.close();

def exe_compress_word(argv):
    word_stat_path, comp_word_stat_path = argv;
    stemmer = EnglishStemmer();
    word_stat = load_word_stat(word_stat_path);
    compress_word_stat = {};
    for word, count in word_stat.items():
        if count <= 0:
            continue;
        word = stemmer.stem(word.lower().decode('utf8'));
        compress_word_stat.__setitem__(word, max(word_stat.get(word,0), count));
    words = compress_word_stat.keys();
    words.sort();
    f = open(comp_word_stat_path, 'w');
    for word in words:
        f.write('%s %d\n' % (word.encode('utf8'), compress_word_stat[word]));
    f.close();

def exe_stat_window(qrel_path, window_db_path):
    window_db = bsddb.hashopen(window_db_path);
    qrel = QRelFile(qrel_path);
    sentence_nums = [];
    sentence_lens = [];
    for q in qrel.keys():
        for d in qrel.get(q).keys():
            if window_db.has_key(d):
                window = window_db[d];
                sentences = window.split('\n');
                sentence_nums.append(len(sentences));
                sentence_lens += map(lambda sentence: len(sentence.split()), sentences);
    print np.mean(sentence_nums), np.median(sentence_nums), np.mean(sentence_lens), np.median(sentence_lens);

if __name__ == '__main__':
    option = sys.argv[1];
    if option == '--extract-text':
        exe_extract_text(*sys.argv[2:]);
    elif option == '--test-extract-text':
        test_extract_text(*sys.argv[2:]);
    elif option == '--extract-window':
        exe_extract_windows(sys.argv[2:]);
    elif option == '--test-extract-window':
        test_extract_windows(sys.argv[2:]);
    elif option == '--extract-word':
        exe_extract_words(sys.argv[2:]);
    elif option == '--extract-topic-word':
        exe_extract_topic_words(sys.argv[2:]);
    elif option == '--compress-word':
        exe_compress_word(sys.argv[2:]);
    elif option == '--stat-window':
        exe_stat_window(*sys.argv[2:]);
    else:
        print 'error param';
    

