from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize;
from nltk.stem.snowball import EnglishStemmer
import sys;


stop_path = 'data/stoplist.dft';

class TextPiece:
    '''
        class of a text piece
    '''
    def __init__(self, text):
        self.text = text.decode('utf8', errors='ignore');

class TextProcessor:
    '''
        abstract class for a text processor;
'''
    def work(self, text_piece):
        pass;

class TextChain(TextProcessor):
    def __init__(self, workers):
        self.workers = workers;

    def work(self, text_piece):
        for worker in self.workers:
            worker.work(text_piece);

class TextTokenizer(TextProcessor):
    '''
        create text.tokens field
    '''
    def __init__(self, tokenize_func):
        self.tokenize_func = tokenize_func;

    def work(self, text_piece):
        text_piece.tokens = self.tokenize_func(text_piece.text);

class TextTokenNormalizer(TextProcessor):
    '''
        normalize to lower case
    '''
    def work(self, text_piece):
        text_piece.tokens = map(lambda token: token.lower(), text_piece.tokens);

class TextStopRemover(TextProcessor):
    '''
        remove the stop words;
    '''
    def __init__(self, stop_path):
        stop_list = map(lambda line: line.strip().lower(), open(stop_path).readlines());
        self.stopwords = set(stop_list);

    def work(self, text_piece):
        text_piece.tokens = filter(lambda token: not self.stopwords.__contains__(token.lower()), text_piece.tokens);

class TextStemmer:
    '''
        stem
    '''
    def __init__(self, stemmer):
        self.stemmer = stemmer;

    def work(self, text_piece):
        text_piece.tokens = map(lambda token: self.stemmer.stem(token), text_piece.tokens);

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




class TextModeler:
    '''
        create a lm field: language model or vector space model
    '''
    def __init__(self, model_factory):
        self.model_factory = model_factory;

    def work(self, text_piece):
        text_piece.lm = self.model_factory.build(text_piece.tokens);



