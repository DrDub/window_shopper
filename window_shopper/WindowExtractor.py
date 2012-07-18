from JudgeFile import QRelFile;
from Index import Index;
from TRECTopics import StandardFormat;
import fastmap;

import bsddb;
import os;
import subprocess;
import sys;
from multiprocessing import Pool;
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize;
from nltk.stem.snowball import EnglishStemmer


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

        f = open(text_path);
        text = ''.join(f.readlines());
        
        os.remove(html_path);
        os.remove(text_path);
        os.remove(title_path);
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

def exe_extract_text(argv):
    judge_path, index_path, text_db_path = argv;
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
        stemmer = EnglishStemmer();
        for term in terms:
            try:
                self.tokens.append(stemmer.stem(term).lower());
            except Exception, e:
                print 'current text:', self.text;
                print 'current term:', term;
                print str(e);
                sys.exit(-1);

def match_window(topic, doc_text):
    topic_terms = set(topic.tokens);
    lines = doc_text.split('\n');
    candidates = [];
    for line in lines:
        sentences = sent_tokenize(line);
        is_candidate = False;
        for sentence in sentences:
            sentence_data = TextPiece(sentence);
            if len(topic_terms.intersection(sentence_data.tokens)) > 0:
                candidates.append(sentence_data);
    return candidates;

def test_extract_windows(argv):
    topic_path, text_path, windows_path = argv;
    topic = TextPiece(open(topic_path).readline().strip());
    doc_text = ''.join(open(text_path).readlines());
    candidates = match_window(topic, doc_text);
    print 'candidates:\n';
    print '\n'.join(map(lambda text_piece: text_piece.text, candidates));

def exe_extract_windows(argv):
    topic_path, judge_path, text_db_path, windows_db_path = argv;
    text_db = bsddb.hashopen(text_db_path);
    window_db = bsddb.hashopen(windows_db_path, 'w');
    judge_file = QRelFile(judge_path);
    topics = StandardFormat().read(topic_path);
    for topic_id, topic_str in topics.items():
        print topic_id;
        sys.stdout.flush();
        topic = TextPiece(topic_str);
        if not judge_file.has_key(topic_id):
            continue;
        docnos = judge_file[topic_id].keys();
        for docno in docnos:
            if not is_cluewebB(docno):
                continue;
            doc_text = text_db[docno];
            window_candidates = match_window(topic, doc_text);
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

def exe_gen_word_db(argv):
    word_stat_path, word_db_path = argv;
    word_db = bsddb.hashopen(word_db_path, 'w');
    lines = open(word_stat_path).readlines();
    for line in lines:
        word, cf, df = line.strip().split();
        word_db[word] = '%s %s' % (cf, df);
    word_db.close();

def load_word_stat(path):
    word_stat = {};
    f = open(path);
    lines = f.readlines();
    word_stat[''] = int(lines[0].split()[0]);
    for line in lines[1:]:
        token, count = line.split();
        word_stat[token] = int(count);
    f.close();
    return word_stat;


def exe_compress_word(argv):
    word_stat_path, comp_word_stat_path = argv;
    stemmer = EnglishStemmer();
    word_stat = load_word_stat(word_stat_path);
    compress_word_stat = {};
    for word, count in word_stat.items():
        word = stemmer.stem(word.decode('utf8'));
        compress_word_stat.__setitem__(word, max(word_stat.get(word,0), count));
    words = compress_word_stat.keys();
    words.sort();
    f = open(comp_word_stat_path, 'w');
    for word in words:
        f.write('%s %d\n' % (word.encode('utf8'), compress_word_stat[word]));
    f.close();

if __name__ == '__main__':
    option = sys.argv[1];
    if option == '--extract-text':
        exe_extract_text(sys.argv[2:]);
    elif option == '--extract-window':
        exe_extract_windows(sys.argv[2:]);
    elif option == '--test-extract-window':
        test_extract_windows(sys.argv[2:]);
    elif option == '--extract-word':
        exe_extract_words(sys.argv[2:]);
    elif option == '--compress-word':
        exe_compress_word(sys.argv[2:]);
    
