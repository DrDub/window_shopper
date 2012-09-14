import numpy as np;
import sys;


def load_snippet_judge(path):
    snippet_judge = {};
    f = open(path);
    lines = f.readlines();
    for line in lines:
        source, topic_id, docno, judge = line.strip().split();
        topic_id = int(topic_id);
        judge = int(judge);
        if not snippet_judge.has_key(source):
            snippet_judge[source] = {};
        if not snippet_judge[source].has_key(topic_id):
            snippet_judge[source][topic_id] = {};
        snippet_judge[source][topic_id][docno] = judge;
    return snippet_judge;


def collect_compare(snippet_judge, doc_judge):
    topic_ids = snippet_judge.keys();
    topic_ids.sort();
    total_matrix = np.zeros((2,2));
    for topic_id in [59]:
        matrix = np.zeros((2,2));
        for docno, snippet_label in snippet_judge[topic_id].items():
            doc_label = int(doc_judge.get_value(topic_id, docno));
            if doc_label > 0:
                if snippet_label > 0:
                    matrix[1,1] += 1;
                else:
                    matrix[1,0] += 1;
            else:
                if snippet_label <= 0:
                    matrix[0,0] += 1;
                else:
                    matrix[0,1] += 1;
        total_matrix += matrix;
    return total_matrix;

def compare_judge(snippet_judge, doc_judge):
    total_matrix = collect_compare(snippet_judge, doc_judge); 
    print 'accuracy:', (total_matrix[0,0] + total_matrix[1,1])/np.sum(total_matrix);
    print 'precision:', total_matrix[1,1]/(total_matrix[0,1]+total_matrix[1,1]);
    recall = total_matrix[1,1]/(total_matrix[1,0]+total_matrix[1,1]);
    print 'recall:', recall;
    print 'positve-agreement:', 2 * total_matrix[1,1]/(total_matrix[0,1]+total_matrix[1,0]+2*total_matrix[1,1]);
    neg_recall = total_matrix[0,0]/(total_matrix[0,1]+total_matrix[0,0]);
    print 'negative-reall:', neg_recall;
    print 'negative-agreement:', 2 * total_matrix[0,0]/(2 * total_matrix[0,0]+total_matrix[1,0]+total_matrix[0,1]);
    print 'average recalls:', (recall + neg_recall)/2;
    print 'geometric average recalls:', np.sqrt(recall * neg_recall);
    print 'perceived relevant:', (total_matrix[0,1] + total_matrix[1,1])/np.sum(total_matrix);
    print 'intrinsic relevant:', (total_matrix[1,0] + total_matrix[1,1])/np.sum(total_matrix);
    print '-' * 20;        

def exe_sign_test(doc_judge_path, snippet_judge_path):
    from JudgeFile import QRelFile;
    from significant_test import t_test_greater;
    snippet_judge = load_snippet_judge(snippet_judge_path);
    doc_judge = QRelFile(doc_judge_path);
    bing_judge = snippet_judge['bing'];
    oq_judge = snippet_judge['windowshop.oq'];
    bing_matrix = collect_compare(bing_judge, doc_judge);
    oq_matrix = collect_compare(oq_judge, doc_judge);
    scores1 = bing_matrix[1]
    scores2 = oq_matrix[1]
    print t_test_greater(scores1, scores2);

def exe_compare(doc_judge_path, snippet_judge_path):
    '''
        compare the doc judgement and snippet judgement;
    '''
    from JudgeFile import QRelFile;
    snippet_judge = load_snippet_judge(snippet_judge_path);
    doc_judge = QRelFile(doc_judge_path);
    for source, source_snippet_judge in snippet_judge.items():
        print source;
        compare_judge(source_snippet_judge, doc_judge);

def test_load(path):
    snippet_judge = load_snippet_judge(path);
    print snippet_judge.keys();

def exe_agree(snippet_judge_path1, snippet_judge_path2):
    '''
        get the agreement of two judgement files
    '''
    n_agree, n_total = 0., 0.;
    values = np.zeros((2,2));
    snippet_judge1 = load_snippet_judge(snippet_judge_path1);
    snippet_judge2 = load_snippet_judge(snippet_judge_path2);
    for source, source_snippet_judge1 in snippet_judge1.items():
        for topic_id, doc_snippet_judge1 in source_snippet_judge1.items():
            for docno, label1 in doc_snippet_judge1.items():
                label2 = snippet_judge2[source][topic_id][docno];
                values[label1, label2] += 1;
    pa = (values[0,0] + values[1,1]) / np.sum(values);
    print 'agreement ratio:', pa;
    p1 = np.sum(values[1]) / np.sum(values);
    p2 = np.sum(values[:,1])/ np.sum(values);
    pe = p1 * p2 + (1-p1) * (1-p2);
    print 'kappa:', (pa-pe)/(1-pe), 

def write_snippet_judge(snippet_judge, path):
    writer = open(path, 'w');
    for source, source_snippet_judge in snippet_judge.items():
        for topic_id, doc_snippet_judge in source_snippet_judge.items():
            for docno, label in doc_snippet_judge.items():
                writer.write('%s %d %s %d\n' % (source, topic_id, docno, label));
    writer.close();

def vote(labels):
    label_counts = {};
    map(lambda label: label_counts.__setitem__(label, label_counts.get(label, 0) + 1), labels);
    label_counts = label_counts.items();
    label_counts.sort(key=lambda label_count: label_count[1], reverse=True);
    label = label_counts[0][0];
    #print labels, label_counts, label;
    return label;

def exe_merge(out_path, *snippet_judge_paths):
    aggregate_snippet_judge = {}; 
    for snippet_judge_path in snippet_judge_paths:
        snippet_judge = load_snippet_judge(snippet_judge_path);
        for source, source_snippet_judge in snippet_judge.items():
            if not aggregate_snippet_judge.has_key(source):
                aggregate_snippet_judge[source] = {};
            for topic_id, doc_snippet_judge in source_snippet_judge.items():
                if not aggregate_snippet_judge[source].has_key(topic_id):
                    aggregate_snippet_judge[source][topic_id] = {};
                for docno, label in doc_snippet_judge.items():
                    if not aggregate_snippet_judge[source][topic_id].has_key(docno):
                        aggregate_snippet_judge[source][topic_id][docno] = [];
                    aggregate_snippet_judge[source][topic_id][docno].append(label);

    for source, source_snippet_judge in aggregate_snippet_judge.items():
        for topic_id, doc_snippet_judge in source_snippet_judge.items():
            for docno, labels in doc_snippet_judge.items():
                aggregate_snippet_judge[source][topic_id][docno] = vote(labels);

    write_snippet_judge(aggregate_snippet_judge, out_path);

oq_importance1 = [ 0.69757288,  0.5813443 ,  0.43045431,  0.8788249,   0.68462119,  0.50352334,  0.36103834,  0.2438666,   0.00841489,  0.08801092 , 0.05770822,  1.,  0.72762308,  0.61077766,  0.85150854];
oq_importance2 = [ 0.66281354, 0.64129034,  0.38344327,  0.621503,    0.57970651,  0.46194554,  0.27413842 , 0.2118412 ,  0.01069773,  0.09348379 , 0.04807041,  1.,  0.86077879 , 0.47460455,  0.58045842]
qe_importance1 = [ 0.4273134 ,  0.35735453  ,0.29936251,  0.46682722 , 0.62030147 , 0.5100286,  0.15712244,  0.26950866,  0.01116927 , 0.08295363 , 0.0406045 ,  0.34034024,  1.  ,        0.91627002 , 0.86129212];
qe_importance2 = [ 0.51024341 , 0.36803769 , 0.32010982 , 0.48549322 , 0.33137504,  0.40569789,   0.17785144 , 0.18976203 , 0.01927949 , 0.04748487 , 0.03082119 , 0.59435948,   0.51731607 , 0.84028565 , 1.        ]

def exe_feature():
    import numpy as np;
    category_idx = [[0,1,2],[3,4,5,6],[7,11,13],[12,14]];
    for importance in [oq_importance1, oq_importance2, qe_importance1, qe_importance2]:
        for i in xrange(len(category_idx)):
            cate_imp = map(lambda idx: importance[idx], category_idx[i]);
            print i, np.mean(cate_imp), max(cate_imp)/max(importance) * .564, sum(cate_imp)/sum(importance);
        print '-' * 20;

def exe_example(snippet_judge_path, doc_judge_path):
#def exe_example(snippet_judge_path, doc_judge_path, bing_path, sum_path, dsm_path):
    from JudgeFile import QRelFile;
    snippet_judge = load_snippet_judge(snippet_judge_path);
    doc_judge = QRelFile(doc_judge_path);
    sources = snippet_judge.keys();
    for topic_id in snippet_judge[sources[0]].keys():
        for docno in snippet_judge[sources[0]][topic_id]:
            in_rel = int(doc_judge.get_value(topic_id, docno));
            if in_rel <= 0:
                in_rel = 0;
            elif in_rel > 0:
                in_rel = 1;
            if snippet_judge['bing'].has_key(topic_id) and  snippet_judge['pablo.short'].has_key(topic_id) and  snippet_judge['windowshop.oq'].has_key(topic_id) :
                bing_per_rel = int(snippet_judge['bing'][topic_id][docno]);
                sum_per_rel = int(snippet_judge['pablo.short'][topic_id][docno]);
                oq_per_rel = int(snippet_judge['windowshop.oq'][topic_id][docno]);
                if (in_rel <> bing_per_rel or in_rel <> sum_per_rel) and in_rel == oq_per_rel == 1:
                    print topic_id, docno, in_rel, bing_per_rel, sum_per_rel, oq_per_rel;
                    #sys.exit(0);


if __name__ == '__main__':
    option = sys.argv[1];
    argv = sys.argv[2:];
    if option == '--test-load':
        test_load(*argv);
    elif option == '--compare':
        exe_compare(*argv);
    elif option == '--agree':
        exe_agree(*argv);
    elif option == '--merge':
        exe_merge(*argv);
    elif option == '--feature':
        exe_feature();
    elif option == '--example':
        exe_example(*argv);
    elif option == '--sign-test':
        exe_sign_test(*argv);
    else:
        print 'error param';
