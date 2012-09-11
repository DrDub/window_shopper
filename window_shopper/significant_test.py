def t_test_greater(scores1, scores2):
    import numpy as np;
    import scipy.stats as stats;
    new_scores1 , new_scores2 = scores1, scores2;
    #for i in xrange(len(scores1)):
        #if scores1[i] == 0 and scores2[i] == 0:
            #continue;
        #new_scores1.append(scores1[i]);
        #new_scores2.append(scores2[i]);
    n = len(scores1);
    diff_scores = np.array(new_scores1) - np.array(new_scores2)
    print new_scores1;
    print new_scores2;
    mean = np.mean(diff_scores);
    std = np.std(diff_scores);
    norm_diff = mean/std * np.sqrt(n);
    print diff_scores;
    print mean, std, norm_diff;
    #print map(lambda value: stats.t(len(scores1) - 1).cdf(value), [.25, .5, 1., 2., 3.]);
    return stats.t(len(scores1) - 1).cdf(norm_diff);


