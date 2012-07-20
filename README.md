window_shopper
==============

Finding the best window around a keyword match in a document.

Distributed under the same license as the Lemur Toolkit.


Re-creating the data/ folder
----------------------------

The system needs a data folder with the following data:

* topics.trec: TREC Web track topics. (TODO: add link?)
* qrels.adhoc: TREC Web judgment file indicating the relevance of the
  document.
* window.db: window data set in berkeley db (hash) format, key is the
  document number, and the value is the windows, separated by '\n'.
  This is the output of WindowExtract.py --extract-window.
  To build this file you need text.db described below.
* text.db: text data set in berkeley db (hash) format, key is the
  document number, and value is the extracted text; word.stat.compress
  #word's document frequency, the words are stemmed in EnglishStemmer
  (NLTK).
  This is the output of WindowExtract.py --extract-text.
  To build this file you need the ClueWeb corpus.


Adding Features
---------------

In FeatureExtractor.py add a abcExtractor class with an extract(self,
topic, doc, window) method (duck-typing) then add it to the extractors
list in the extract_window_feature method.

topic, doc and window are all instances of TextPiece as defined in
WindowExtractor.py.