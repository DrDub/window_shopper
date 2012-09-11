'''
Created on Aug 8, 2012

@author: hejing
'''
from sklearn.ensemble.base import BaseEnsemble;
from sklearn.ensemble._gradient_boosting import predict_stages;

from sklearn.utils import check_random_state;

from sklearn.tree.tree import Tree
from sklearn.tree._tree import _find_best_split
from sklearn.tree._tree import _random_sample_mask
from sklearn.tree._tree import _apply_tree
from sklearn.tree._tree import MSE
from sklearn.tree._tree import DTYPE

import numpy as np;
import pylab as pl;
import sys;
import time;

from multiprocessing import Pool;

class ConstEstimator(object):
    def __init__(self, val):
        self.val = val;
        
    def predict(self, X):
        y = np.empty((X.shape[0],1), dtype=np.float64)
        y.fill(self.val)
        return y;
    
class DocPairSampler:
    def __init__(self, random_state):
        self.random_state = random_state;
        
    def sample(self, rd, groups, n, mask_doc_pairs=set()):
        samples = [];
        group_ids = np.array(groups.keys());
        space_size = len(groups);
        sample_num = 0;
        for i in xrange(100000):
            query_index = self.random_state.randint(0, space_size);
            docids = groups[group_ids[query_index]].keys();
            rel_docids = filter(lambda docid: rd[docid] > 0, docids);
            irrel_docids = filter(lambda docid: rd[docid] <= 0, docids);
            if len(rel_docids) == 0 or len(irrel_docids) == 0:
                continue;
            docid1 = rel_docids[self.random_state.randint(0, len(rel_docids))];
            docid2 = irrel_docids[self.random_state.randint(0, len(irrel_docids))];
            if mask_doc_pairs.__contains__((docid1, docid2)):
                continue;
            samples.append((docid1, docid2));
            sample_num += 1;
            if sample_num >= n:
                break;
            
        return samples;

class PredictSortGroups(dict):
    def __init__(self, pred, groups):
        for group in groups.values():
            for docid, senids in group.items():
                self[docid] = senids[pred[senids,0].argsort()[::-1]]; 

#class GradientComputer:
#    def __init__(self, rd, rs, pred, groups, pred_sort_groups, random_state, do_consider_correct):
#        self.rd = rd;
#        self.rs = rs;
#        self.pred = pred;
#        self.groups = groups;
#        self.pred_sort_groups = pred;
#        self.do_consider_correct = do_consider_correct;
#        
#    def __call__(self, docpairs):    

class RankError:
    def __init__(self, n1, n2, n3, tau):
        self.n1 = n1;
        self.n2 = n2;
        self.n3 = n3;
        self.tau = tau;
    
    """Loss function for two-layer ranker. """
    def init_estimator(self):
        return ConstEstimator(1.0)

    def __call__(self, rd, rs, pred, groups, pred_sort_groups, random_state, docpair_samples):
        error = 0;
        for docid1, docid2 in docpair_samples:
            rd1, rd2 = rd[docid1], rd[docid2];
            senid1 = pred_sort_groups.get(docid1)[0];
            senid2 = pred_sort_groups.get(docid2)[0];
            rs1, rs2 = rs[senid1], rs[senid2];
            if (rd1 - rd2) * (rs1 - rs2) <= 0:
#                error += np.abs(rs1 - rs2);
                error += 1.0
        return error / len(docpair_samples);

    def update_terminal_regions(self, tree, X, y, residual, y_pred,
                                sample_mask, learn_rate=1.0, k=0):
        """Least squares does not need to update terminal regions.

        But it has to update the predictions.
        """
#        print y_pred;
        # update predictions
        y_pred[:,0] += learn_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred):
        pass





class NegativeGradientComputer:
    def __init__(self, rd, rs, pred, groups, pred_sort_groups, n_sen_num, tau, do_consider_correct):
        self.rd = rd;
        self.rs = rs;
        self.pred = pred;
        self.pred_sort_groups = pred_sort_groups;
        self.n_sen_num = n_sen_num;
        self.tau = tau;
        self.do_consider_correct = do_consider_correct;
        
    def compute(self, docpairs):
        negative_gradients = np.zeros((2, self.rs.shape[0]), dtype=np.float64);
        
#        p = Pool(5);
        updates = map(self, docpairs);
        for pair_update in updates:
            for senid, update in pair_update:
                negative_gradients[0, senid] += update;
                negative_gradients[1, senid] += 1.0;
                
        negative_gradients, senid_masks = self.aggregate_negative_gradient(negative_gradients);
        return negative_gradients, senid_masks;        
        
    def __call__(self, docpair):
#       
#        print 'updating gradient...'
        
        docid1, docid2 = docpair;
        rd1, rd2 = self.rd[docid1], self.rd[docid2];
        senids1 = self.pred_sort_groups.get(docid1);
        senids2 = self.pred_sort_groups.get(docid2);
        rs1, rs2 = self.rs[senids1[0]], self.rs[senids2[0]];
        update_list = [];
        if (rd1 - rd2) * (rs1 - rs2) <= 0:        
            update_list += self.update_for_error(senids1, rs2); 
            update_list += self.update_for_error(senids2, rs1);
        elif self.do_consider_correct:
            update_list += self.update_for_correct(senids1, rs2);
            update_list += self.update_for_correct(senids2, rs1);
        return update_list;
    
    def aggregate_negative_gradient(self, negative_gradients):
        
        senid_masks = negative_gradients[1, ] <> 0;
#        print 'ng:', negative_gradients;
#        print 'masks:', masks;
        negative_gradients[1, negative_gradients[1, ] == 0] = 1;
        negative_gradients = negative_gradients[0, ] / negative_gradients[1, ];
        return negative_gradients, senid_masks;
        
    
    def update_for_error(self, sen_ids, pilot_rs):
        update_list = [];
        
        rs1 = self.rs[sen_ids];
        first_sen_id = sen_ids[0];
        first_rs = self.rs[first_sen_id];
        if first_rs > pilot_rs:
            good_sen_ids = sen_ids[rs1 < pilot_rs]; 
        else:
            good_sen_ids = sen_ids[rs1 > pilot_rs];
        if good_sen_ids.size == 0:
            return update_list;
        
        sample_good_sen_ids = good_sen_ids[np.random.randint(0, len(good_sen_ids), self.n_sen_num)];
        first_sen_pred = self.pred[first_sen_id];
        for good_sen_id in sample_good_sen_ids:
            good_sen_pred = self.pred[good_sen_id];
#            print 'first sen id=%d, fir_rs=%f, good_sen_id=%d, good_rs=%f' % (first_sen_id, first_rs, good_sen_id, rs[good_sen_id]);
#            print 'first sen pred=%f, good_sen_pred=%f' % (first_sen_pred, good_sen_pred);
#            print 'update for the first sen=%f, update for the good sen=%f' % (good_sen_pred - first_sen_pred - self.tau, first_sen_pred - good_sen_pred + self.tau);
            update_list.append((first_sen_id, good_sen_pred - first_sen_pred - self.tau));
            update_list.append((good_sen_id, first_sen_pred - good_sen_pred + self.tau));
        return update_list;
        
    def update_for_correct(self, sen_ids, pilot_rs):
        update_list = [];
        
        rs1 = self.rs[sen_ids];
        first_sen_id = sen_ids[0]; 
        first_rs = self.rs[first_sen_id];
        if first_rs > pilot_rs:
            bad_sen_ids = sen_ids[rs1 < pilot_rs]; 
        else:
            bad_sen_ids = sen_ids[rs1 > pilot_rs];
        if bad_sen_ids.size == 0:
            return update_list;
        
        sample_bad_sen_ids = bad_sen_ids[np.random.randint(0, len(bad_sen_ids), self.n_sen_num)];
        for bad_sen_id in sample_bad_sen_ids:
#            print first_sen_id, bad_sen_id;
            update_list.append((first_sen_id, 0));
            update_list.append((bad_sen_id, 0));
        return update_list;

    
class GradientBoostingRanker(BaseEnsemble):
    """Abstract base class for Gradient Boosting. """
    def __init__(self, learn_rate, n_estimators, min_samples_split,
                 min_samples_leaf, max_depth, n1, n2, n3, tau, do_consider_correct, random_state):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        self.n_estimators = n_estimators

        if learn_rate <= 0.0:
            raise ValueError("learn_rate must be greater than 0")
        self.learn_rate = learn_rate

        if min_samples_split <= 0:
            raise ValueError("min_samples_split must be larger than 0.")
        self.min_samples_split = min_samples_split

        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be larger than 0.")
        self.min_samples_leaf = min_samples_leaf

        self.n1 = n1;
        self.n2 = n2;
        self.n3 = n3;
        self.tau = tau;

        if max_depth <= 0:
            raise ValueError("max_depth must be larger than 0.")
        self.max_depth = max_depth
        
        self.do_consider_correct = do_consider_correct;

        self.random_state = check_random_state(random_state)

        self.estimators_ = None

    def fit_stage(self, i, X, X_argsorted, rd, rs, pred, groups, pred_sort_groups, training_docpair_samples):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """
        
        k = 0;
#        print 'computing gradient...'
        residual, masks = self.negative_gradient_computer.compute(training_docpair_samples);


#        print 'building tree...';
        # induce regression tree on residuals
        tree = Tree(1, self.n_features)
        tree.build(X, residual, MSE(), self.max_depth,
                       self.min_samples_split, self.min_samples_leaf, 0.0,
                       self.n_features, self.random_state, _find_best_split,
                       masks, X_argsorted)

        # update tree leaves and pred
        self.loss_.update_terminal_regions(tree, X, 0, residual, pred,
                                               masks, self.learn_rate,
                                               k=k)

        # add tree to ensemble
        self.estimators_[i, k] = tree

        return pred

    def fit(self, X, rd, rs, groups, test_X=None, test_rd=None, test_rs=None, test_groups=None):
        """Fit the gradient boosting model for two-layer ranking problem.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features. Use fortran-style
            to avoid memory copies.

        rd : array-like, shape = [n_docs]
            Relevance Judgment for docs
            
        rs: array-like, shape = [n_samples]
            Retrieval Function Score for sentences
            
        groups: a list
            Each element of which is a map of doc id->sentence ids
            It describes query-doc-sentence relationship

        Returns
        -------
        self : object
            Returns self.
        """
#        print 'initial...';
        X = np.asfortranarray(X, dtype=DTYPE)
        rd = np.ascontiguousarray(rd)
        rs = np.ascontiguousarray(rs);

        n_samples, n_features = X.shape
        if rs.shape[0] != n_samples:
            raise ValueError("Number of labels does not match " \
                             "number of samples.")
        self.n_features = n_features

        loss = RankError(self.n1, self.n2, self.n3, self.tau);

        # store loss object for future use
        self.loss_ = loss
        

        self.init = ConstEstimator(0);

        # create argsorted X for fast tree induction
        X_argsorted = np.asfortranarray(
            np.argsort(X.T, axis=1).astype(np.int32).T)

        # sampling training and test
        docpair_sampler = DocPairSampler(self.random_state);
#        print 'generating training...'
        

        # init predictions
        pred = self.init.predict(X)

        self.estimators_ = np.empty((self.n_estimators, 1),
                                    dtype=np.object)

        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if test_X <> None:
            test_X = np.asfortranarray(test_X, dtype=DTYPE);
            test_rd = np.ascontiguousarray(test_rd);
            test_rs = np.ascontiguousarray(test_rs);
            test_y_pred = self.init.predict(test_X);
            self.oob_score_ = np.zeros((self.n_estimators), dtype=np.float64);
        else:
            self.oob_score_ = 0;

        t = time.time();
        pred_sort_groups = PredictSortGroups(pred, groups);
        training_docpair_samples = docpair_sampler.sample(rd, groups, self.n1);
        test_docpair_samples = docpair_sampler.sample(rd, groups, self.n3);
#        print 'training-score:', loss(rd, rs, pred, groups, pred_sort_groups, self.random_state, training_docpair_samples);
#        print 'test-score:', loss(rd, rs, pred, groups, pred_sort_groups, self.random_state, test_docpair_samples);
        # perform boosting iterations
        for i in range(self.n_estimators):
            print '%.4f' % (time.time() - t), 'iteration', i;
            self.negative_gradient_computer = NegativeGradientComputer(rd, rs, pred, groups, pred_sort_groups, self.n2, self.tau, self.do_consider_correct);
#            print 'sorting predict value...'

            training_docpair_samples = docpair_sampler.sample(rd, groups, self.n1);
            

#            print 'fitting...'
            # fit next stage of trees
            pred = self.fit_stage(i, X, X_argsorted, rd, rs, pred, groups, pred_sort_groups, training_docpair_samples);

            pred_sort_groups = PredictSortGroups(pred, groups);
#            print 'computing loss...'
            # track deviance (= loss)
            self.train_score_[i] = loss(rd, rs, pred, groups, pred_sort_groups, self.random_state, training_docpair_samples);
            print 'training score:', self.train_score_[i];
    
            if test_X <> None:
                test_docpair_samples = docpair_sampler.sample(test_rd, test_groups, self.n3);
                test_y_pred[:,0] += self.learn_rate * self.estimators_[i, 0].predict(test_X).ravel();
                test_pred_sort_groups = PredictSortGroups(test_y_pred, test_groups);
                self.oob_score_[i] = loss(test_rd, test_rs, test_y_pred, test_groups, test_pred_sort_groups, self.random_state, test_docpair_samples);
                print 'test score:', self.oob_score_[i]
            
        return self

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()
    
    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples]
            The predicted values.
        """
        X = np.atleast_2d(X)
        X = X.astype(DTYPE)
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, " \
                             "call `fit` before `predict`.")
        if X.shape[1] != self.n_features:
            raise ValueError("X.shape[1] should be %d, not %d." % 
                             (self.n_features, X.shape[1]))

        y = self.init.predict(X).astype(np.float64)
#        print self.estimators_.shape, X.shape, y.shape;
        predict_stages(self.estimators_, X, self.learn_rate, y)
        return y




    @property
    def feature_importances_(self):
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, " \
                             "call `fit` before `feature_importances_`.")
        total_sum = np.zeros((self.n_features,), dtype=np.float64)
        for stage in self.estimators_:
            stage_sum = sum(tree.compute_feature_importances(method='squared')
                            for tree in stage) / len(stage)
            total_sum += stage_sum

        importances = total_sum / len(self.estimators_)
        return importances

    def staged_decision_function(self, X):
        """Compute decision function for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        f : array of shape = [n_samples, n_classes]
            The decision function of the input samples. Classes are
            ordered by arithmetical order. Regression and binary
            classification are special cases with ``n_classes == 1``.
        """
#        X = np.atleast_2d(X)
#        X = X.astype(DTYPE)
#
#        if self.estimators_ is None or len(self.estimators_) == 0:
#            raise ValueError("Estimator not fitted, call `fit` " \
#                             "before `staged_decision_function`.")
#        if X.shape[1] != self.n_features:
#            raise ValueError("X.shape[1] should be %d, not %d." % 
#                             (self.n_features, X.shape[1]))
#
#        score = self.init.predict(X).astype(np.float64)
#
#        for i in range(self.n_estimators):
#            predict_stage(self.estimators_, i, X, self.learn_rate, score)
#            yield score


def filter_datset(inpath, outpath, n):
    reader = open(inpath);
    writer = open(outpath, 'w');
    n = int(n);
    line = reader.readline();
    while line:
        if len(line.strip().split()[-1].split(',')) == n:
            writer.write(line);
        line = reader.readline();
    reader.close();
    writer.close();

def load_dataset(path):
        f = open(path);
        X = [];
        sr_array = [];
        dr_array = [];
        groups = {};
        lines = f.readlines();
        
        next_doc_id = 0;
        sen_id = -1;
        docno_docid_map = {};
        for line in lines:
            line = line.strip();
            pos = line.find('#');
            if pos >= 0:
                line = line[:pos];
            tokens = line.split();
            qid, docno, dr, sr = tokens[:4];
            qid, dr = int(qid), int(dr);
            sr = float(sr);
            features = map(float, tokens[4].split(','));
            if len(features) < 15:
                continue;
            
            if not groups.has_key(qid):
                groups[qid] = {};
            if not docno_docid_map.has_key(docno):
                docno_docid_map[docno] = next_doc_id;
                dr_array.append(dr);
                next_doc_id += 1;
            doc_id = docno_docid_map[docno];
            if not groups[qid].has_key(doc_id):
                groups[qid][doc_id] = [];
                
            sen_id += 1;
            groups[qid][doc_id].append(sen_id);
            
            X.append(features);
            sr_array.append(sr);
            
        X = np.array(X);
        dr_array = np.array(dr_array);
        sr_array = np.array(sr_array);
        for group in groups.values():
            for doc_id, sen_ids in group.items():
                group[doc_id] = np.array(sen_ids);
        return X, dr_array, sr_array, groups;
            

def write_predict(pred_y, path):
    n_row = pred_y.shape[0];
    f = open(path, 'w');
    for i in xrange(n_row):
        f.write('%f\n' % pred_y[i,0]);
    f.close();

def do_rank(argv):
    path, test_path = argv;
    
    params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 1,
              'min_samples_leaf':1, 'random_state':None, 'do_consider_correct':1,
          'learn_rate': 0.2, 'n1': 10000, 'n2': 1, 'n3': 20000, 'tau': 0.01};

    ranker = GradientBoostingRanker(**params);
    
    print 'loading data...'
    X, dr, sr, groups = load_dataset(path)
    test_X, test_dr, test_sr, test_groups = load_dataset(test_path);
    
    print 'starting fit...'
    ranker.fit(X, dr, sr, groups, test_X, test_dr, test_sr, test_groups);
#    ranker.fit(X, dr, sr, groups);

#    print ranker.train_score_;
    pl.figure(figsize=(12, 6))
    pl.subplot(1, 2, 1)
    pl.title('Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, ranker.train_score_, 'b-',
        label='Training Set Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, ranker.oob_score_, 'r-',
        label='Test Set Deviance')
    pl.legend(loc='upper right')
    pl.xlabel('Boosting Iterations')
    pl.ylabel('Deviance')

    # Plot feature importance
    feature_importance = ranker.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1, 2, 2)
    pl.barh(pos, feature_importance[sorted_idx], align='center')
    pl.yticks(pos, np.array(range(len(feature_importance))));
    pl.xlabel('Relative Importance')
    pl.title('Variable Importance')
    
    print feature_importance;

    #pl.show();
    
    
def do_label(train_path, test_path, pred_path):
    params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 1,
              'min_samples_leaf':1, 'random_state':None, 'do_consider_correct':1,
          'learn_rate': 0.2, 'n1': 10000, 'n2': 1, 'n3': 20000, 'tau': 0.01};

    ranker = GradientBoostingRanker(**params);
    
    print 'loading data...'
    X, dr, sr, groups = load_dataset(train_path)
    test_X, test_dr, test_sr, test_groups = load_dataset(test_path);
    
    print 'starting fit...'
    ranker.fit(X, dr, sr, groups, test_X, test_dr, test_sr, test_groups);
    
    pred_y = ranker.predict(test_X);
    write_predict(pred_y, pred_path);

    
def do_parameter_selection(argv):
    path, test_path = argv;
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
              'min_samples_leaf':1, 'random_state':None, 'do_consider_correct':1,
          'learn_rate': 0.2, 'n1': 2000, 'n2': 1, 'tau': 0.01};
    print 'loading data...'
    X, dr, sr, groups = load_dataset(path)
    test_X, test_rd, test_rs, test_groups = load_dataset(test_path);
    
#    test_X = np.asfortranarray(test_X, dtype=DTYPE);
    test_rd = np.ascontiguousarray(test_rd);
    test_rs = np.ascontiguousarray(test_rs);
    test_docpair_samples = DocPairSampler(np.random.RandomState()).sample(test_rd, test_groups, 20000);
    
    from sklearn.grid_search import IterGrid;
    param_grid = IterGrid({'n_estimators':[200,400,600,800,1000], 'n1':[1000,2000,5000], 'learn_rate':[.1,.2,.3] });
    for param in param_grid:
        print param;
        params.update(param);
        ranker = GradientBoostingRanker(**params);
        ranker.fit(X, dr, sr, groups);
        test_y_pred = ranker.predict(test_X);
        test_pred_sort_groups = PredictSortGroups(test_y_pred, test_groups);
        test_loss = ranker.loss_(test_rd, test_rs, test_y_pred, test_groups, test_pred_sort_groups, ranker.random_state, test_docpair_samples);
        print ranker.train_score_[-1], test_loss;
    
    
    
def do_convert_standard_feature(original_feature_path, standard_feature_path):
    '''
        convert 
    '''
    in_path, out_path = argv;
    writer = open(out_path, 'w');
    lines = open(in_path).readlines();
    for line in lines:
        qid, docno, rd, sid, features = line.strip().split();
        rs = features.split(',')[-2];
        writer.write('%s %s %s %s %s\n' % (qid, docno, rd, rs, features));
    writer.close();
    
    
if __name__ == '__main__':
    option = sys.argv[1];
    argv = sys.argv[2:];
    if option == '--convert':
        do_convert_standard_feature(*argv);
    elif option == '--fit':
        do_rank(argv);
    elif option == '--param':
        do_parameter_selection(argv);
    elif option == '--filter':
        filter_datset(*argv);
    elif option == '--label':
        do_label(*argv);
    else:
        print 'error param!';
    
    
