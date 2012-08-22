import sys

from JudgeFile import QRelFile
from TRECTopics import StandardFormat
from TrainGenerator import *;


topics = StandardFormat().read(sys.argv[1]);
judge_file = QRelFile(sys.argv[2])
lemmas = set()
topic_ids = judge_file.keys()
for topic_id in topic_ids:
    if not topics.has_key(topic_id):
            continue
    topic_str = topics[topic_id]
    topic = TextPiece(topic_str)

    lemmas.update(topic.lemmas)

for lemma in lemmas:
    print lemma
