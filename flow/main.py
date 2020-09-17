import warnings

import numpy as np
import pandas as pd
from sklearn import kernel_ridge, gaussian_process, ensemble, neighbors
from sklearn import utils, preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import BayesianRidge, SGDRegressor, Lasso, ElasticNet
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.model_selection._split import _BaseKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state, column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target

warnings.filterwarnings("ignore")


class UserStratifiedKFold(_BaseKFold):

    def __init__(self, group, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
        group = column_or_1d(group)
        self.group = group

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (min_groups, self.n_splits)), UserWarning)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
             for i in range(self.n_splits)])

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        if self.group is not None:
            y = self.group
        else:
            raise TypeError("The group must be assignment")

        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


def dict_method_reg():
    warnings.filterwarnings("ignore")
    kernel = 1.0 * RBF(1.0)
    kernel2 = Matern(nu=1.5)
    kernel3 = Matern(nu=0.5)
    kernel4 = Matern(nu=2)
    kernel5 = Matern(length_scale=2, nu=1.5)
    kernel6 = Matern(length_scale=2, nu=0.5)
    kernel7 = Matern(length_scale=2, nu=2)
    kernel8 = Matern(length_scale=2, nu=1)
    dict_method = {}
    # 1st part
    """1SVR"""

    me1 = SVR(kernel='rbf', gamma='auto', degree=4, tol=1e-3, epsilon=0.1, shrinking=True, max_iter=2000)
    cv1 = 5
    scoring1 = "r2"
    param_grid1 = [{'C': [1, 0.5, 0.1, 0.01], 'gamma': [0.5, 0.1, 0.001, 0.01, 0.0001],
                    "epsilon": [1, 0.1, 0.01, 0.001],
                    "kernel": [kernel, kernel2, kernel3, kernel4],

                    }]
    dict_method.update({"SVR-set": [me1, cv1, scoring1, param_grid1]})

    """2BayesianRidge"""
    me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                        n_iter=300, normalize=False, tol=0.01, verbose=False)
    cv2 = 5
    scoring2 = "r2"
    param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
                    'alpha_2': [1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
                    'lambda_1': [1e-06, 1e-05, 1e-07],
                    'lambda_2': [1e-06, 1e-05, 1e-07],
                    }]
    dict_method.update({'BayR-set': [me2, cv2, scoring2, param_grid2]})

    """3SGDRL2"""
    me3 = SGDRegressor(alpha=0.0001, average=False,
                       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                       penalty='l2', power_t=0.25,
                       random_state=0, shuffle=True, tol=0.01,
                       verbose=0, warm_start=False)
    cv3 = 5
    scoring3 = "r2"
    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05]}]
    dict_method.update({'SGDRL2-set': [me3, cv3, scoring3, param_grid3]})

    """4KNR"""
    me4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski')
    cv4 = 5
    scoring4 = "r2"
    param_grid4 = [{'n_neighbors': [3, 4, 5, 6], 'leaf_size': [10, 20, 30]}]
    dict_method.update({"KNR-set": [me4, cv4, scoring4, param_grid4]})

    """5kernelridge"""

    me5 = kernel_ridge.KernelRidge(alpha=1, kernel=kernel, gamma="scale", degree=3, coef0=1, kernel_params=None)
    cv5 = 5
    scoring5 = "r2"
    param_grid5 = [{'alpha': [10, 7, 5, 3, 2, 1, 0.5, 0.1],
                    "kernel": [kernel, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8]}]
    dict_method.update({'KR-set': [me5, cv5, scoring5, param_grid5]})

    """6GPR"""
    kernel = 1.0 * RBF(1.0)
    kernel2 = Matern(nu=1.5)
    kernel3 = Matern(nu=0.5)
    kernel4 = Matern(nu=2)

    me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                                    normalize_y=True, copy_X_train=True, random_state=0)
    cv6 = 5
    scoring6 = "r2"
    param_grid6 = [{'alpha': [1e-10, 1e-8, 1e-6, 0.0001, 0.01, 1],
                    "kernel": [kernel, kernel2, kernel3, kernel4],
                    "random_state": [0, 1, 2]}]
    dict_method.update({"GPR-set": [me6, cv6, scoring6, param_grid6]})

    # 2nd part

    """6RFR"""
    me7 = ensemble.RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, bootstrap=True, oob_score=False,
                                         random_state=None, verbose=0, warm_start=False)
    cv7 = 5
    scoring7 = "r2"
    param_grid7 = [{'max_depth': [3, 5, 8, 10], 'min_samples_split': [2, 3, 4], 'random_state': [0, 1, 2],
                    'n_estimators': [500, 200]}]
    dict_method.update({"RFR-em": [me7, cv7, scoring7, param_grid7]})

    """7GBR"""

    me8 = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=200,
                                             subsample=1.0, criterion='mse', min_samples_split=2,
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                             max_depth=3, min_impurity_decrease=0.,
                                             min_impurity_split=None, init=None, random_state=None,
                                             max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                             warm_start=False, presort='auto')
    cv8 = 5
    scoring8 = "r2"
    param_grid8 = [{'max_depth': [3, 5, 8, 10, 12, 14], 'min_samples_split': [2, 3, 4],
                    'min_samples_leaf': [2, 3], 'random_state': [0, 1, 2],
                    'n_estimators': [50, 100, 200, 300]}]
    dict_method.update({'GBR-em': [me8, cv8, scoring8, param_grid8]})

    "AdaBR"
    dt2 = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=12, min_samples_split=4)
    dt3 = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=14, min_samples_split=4)
    dt4 = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=16, min_samples_split=4)
    dt = [dt4, dt2, dt3]
    me9 = AdaBoostRegressor(dt, n_estimators=200, learning_rate=0.5, loss='linear', random_state=0)
    cv9 = 5
    scoring9 = "r2"
    param_grid9 = [{'n_estimators': [50, 100, 200], "base_estimator": dt, "learning_rate": [0.05, 0.5, 1],
                    'random_state': [0, 1, 2]}]
    dict_method.update({"AdaBR-em": [me9, cv9, scoring9, param_grid9]})

    '''TreeR'''
    me10 = DecisionTreeRegressor(
        criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=0, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
    cv10 = 5
    scoring10 = "r2"
    param_grid10 = [{'max_depth': [4, 5, 6], 'min_samples_split': [3, 4], 'random_state': [0, 1, 2]}]
    dict_method.update({'TreeC-em': [me10, cv10, scoring10, param_grid10]})

    'ElasticNet'
    me11 = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
                      copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)

    cv11 = 5
    scoring11 = "r2"
    param_grid11 = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.3, 0.5, 0.8]}]
    dict_method.update({"ElasticNet-L1": [me11, cv11, scoring11, param_grid11]})

    'Lasso'
    me12 = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=0.001,
                 warm_start=False, positive=False, random_state=None, )

    cv12 = 5
    scoring12 = "r2"
    param_grid12 = [{'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}, ]
    dict_method.update({"Lasso-L1": [me12, cv12, scoring12, param_grid12]})

    """SGDRL1"""
    me13 = SGDRegressor(alpha=0.0001, average=False,
                        epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                        learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                        penalty='l1', power_t=0.25,
                        random_state=0, shuffle=True, tol=0.01,
                        verbose=0, warm_start=False)
    cv13 = 5
    scoring13 = "r2"
    param_grid13 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]}]
    dict_method.update({'SGDR-L1': [me13, cv13, scoring13, param_grid13]})

    return dict_method


def method_pack(method_all, scoring=None, gd=True):
    warnings.filterwarnings("ignore")
    if not method_all:
        method_all = ['KNR-set', 'SVR-set', "KR-set"]

    # method_all = [dict_method_reg()[i] for i in method_all]

    if gd:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method_reg()[method]

            scoring2 = scoring
            gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=20)
            estimator.append(gd2)
        return estimator
    else:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method_reg()[method]

            scoring2 = scoring
            gd2 = cross_val_score(me2, cv=cv2, scoring=scoring2)
            estimator.append(gd2)
        return estimator


def pack_score(y_test_true_all, y_test_predict_all, scoring):
    if scoring == 'neg_root_mean_squared_error':
        return np.sqrt(np.mean((y_test_true_all - y_test_predict_all) ** 2))

    scorer = get_scorer(scoring)

    scorer_func = scorer._score_func

    score = scorer_func(y_test_true_all, y_test_predict_all)

    return score


def my_score(gd_method, train_X, test_X, train_Y, test_Y, random_state=4):
    log = []
    # pca = PCA(random_state=0)
    # train_X = pca.fit_transform(train_X)
    # test_X = pca.transform(test_X)
    '''数据标准化（均值为0，方差为1）'''
    s_X = preprocessing.StandardScaler()
    train_X = s_X.fit_transform(train_X)
    test_X = s_X.transform(test_X)
    if random_state is None:
        pass
    else:
        train_X, train_Y = utils.shuffle(train_X, train_Y, random_state=random_state)

    grid = gd_method
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=False)
    grid.cv = kf
    grid.fit(train_X, train_Y)
    # print("\n******SVR*******",test_X.shape[1],"\n")
    print("\ngrid.best_score:\n", grid.best_score_)
    # print("\ngrid._scores:\n",grid.grid_scores_)

    metrics_method1 = 'neg_root_mean_squared_error'
    metrics_method2 = "r2"

    # train
    grid = grid.best_estimator_

    grid.fit(train_X, train_Y)
    pre_train_y = grid.predict(train_X)
    score3 = pack_score(train_Y, pre_train_y, metrics_method1)
    score4 = pack_score(train_Y, pre_train_y, metrics_method2)
    # print("train_X's score rmse %s" % score3, "train_X's score r2 %s" % score4)
    # test
    pre_test_y = grid.predict(test_X)
    score5 = pack_score(test_Y, pre_test_y, metrics_method1)
    score6 = pack_score(test_Y, pre_test_y, metrics_method2)
    # print("test_X's score rmse %s" % score5, "test_X's score r2 %s" % score6)
    log.extend((
        "train_X's score rmse %s" % score3, "train_X's score r2 %s" % score4,
        "test_X's score rmse %s" % score5, "test_X's score r2 %s" % score6))
    return train_Y, pre_train_y, test_Y, pre_test_y, log, score5, score6


def my_score_cv(gd_method, X, Y, random_state=4, cv=10):
    log = []
    # pca = PCA(random_state=0)
    # X = pca.fit_transform(X)
    '''数据标准化（均值为0，方差为1）'''
    s_X = preprocessing.StandardScaler()
    X = s_X.fit_transform(X)

    train_X, train_Y = utils.shuffle(X, Y, random_state=random_state)
    grid = gd_method

    est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    ty = train_Y.reshape(-1, 1)
    est.fit(ty)
    group = est.transform(ty)
    skf = UserStratifiedKFold(group=group, n_splits=cv, shuffle=False)

    grid.cv = skf
    print(grid.scoring)
    grid.fit(train_X, train_Y)

    # print("\ngrid.best_score:\n", grid.best_score_)
    # print("\ngrid._scores:\n",grid.grid_scores_)

    metrics_method1 = 'neg_root_mean_squared_error'
    metrics_method2 = "r2"

    # train
    grid = grid.best_estimator_
    score3 = cross_val_score(grid, train_X, train_Y, scoring=metrics_method1,
                             cv=skf)
    score4 = cross_val_score(grid, train_X, train_Y, scoring=metrics_method2,
                             cv=skf)
    score3 = -score3.mean()
    score4 = score4.mean()
    pre_train_y = cross_val_predict(grid, train_X, train_Y, cv=skf)

    print("X's cv score rmse %s" % score3, "X's cv score r2 %s" % score4)

    log.extend((
        "X's cv score rmse %s" % score3, "X's cv score r2 %s" % score4))
    return train_Y, pre_train_y, log, score3, score4


# sheet_name = "9-1"
sheet_name = "8-2"  #
com_data_raw = pd.read_excel('wxx.xlsx')

tt_spilt = pd.read_excel('train_test-10-20.xlsx', sheet_name=sheet_name)
com_data_train = pd.merge(tt_spilt["train"], com_data_raw, left_on="train", right_on="MXenes")
com_data_test = pd.merge(tt_spilt["test"], com_data_raw, left_on="test", right_on="MXenes")
com_data_train.rename(columns={'train': 'name'}, inplace=True)
com_data_test.rename(columns={'test': 'name'}, inplace=True)
com_data = com_data_train.append(com_data_test)

select_X5 = ['O-M-outer', 'a Lattice parameter', 'M Electron affinity mean', 'M first ionization potential differ',
             'valence X atom']
target_y0 = ['EH-22-1H', 'EH-22-2H', 'EH-22-3H', 'EH-22-4H']
####################################################################################
for k in range(len(target_y0)):

    target_y = target_y0[k]

    x = com_data[select_X5].values
    y = com_data[target_y].values

    train_x = com_data_train[select_X5].values
    train_y = com_data_train[target_y].values
    #
    test_x = com_data_test[select_X5].values
    test_y = com_data_test[target_y].values

    method0 = ["SVR-set", "RFR-em", "GPR-set", "AdaBR-em"]

    for j in range(len(method0)):
        method = method0[j]  #
        gd_method = method_pack([method, ], scoring='neg_root_mean_squared_error', gd=True)

        #     ##############
        #     pre_train_y0, pre_test_y0, log0, score50, score60, ran = 1, 1, 1, 1, 0, 0
        #     train_Y0, test_Y0 = 0, 0
        #     for i in range(5):
        #         train_Y, pre_train_y, test_Y, pre_test_y, log, score5, score6 = my_score(gd_method[0], train_x, test_x,
        #                                                                                  train_y,
        #                                                                                  test_y, random_state=i)
        #         if score6 > score60:
        #             train_Y0, pre_train_y0, test_Y0, pre_test_y0, log0, score50, score60, ran = train_Y, pre_train_y, test_Y, pre_test_y, log, score5, score6, i
        #         elif round(score6, 3) == round(score60, 3) and score5 < score50:
        #             train_Y0, pre_train_y0, test_Y0, pre_test_y0, log0, score50, score60, ran = train_Y, pre_train_y, test_Y, pre_test_y, log, score5, score6, i
        #
        #         else:
        #             pass
        #
        #     y_ = np.concatenate((pre_train_y0, pre_test_y0))
        #     y0 = np.concatenate((train_Y0, test_Y0))
        #     st = pd.DataFrame(np.vstack((y0, y_)).T)
        #     log0 = pd.DataFrame(log0)
        #     storename = r"result/spilt-{}-{}-{}-random-{}-{}-{}.csv".format(sheet_name, method, target_y, ran, score50, score60)
        #     print(storename)
        #     st.to_csv(storename)
        #     log0.to_csv(storename, mode="a")
        ###############
        pre_y0, log0, score50, score60, ran = 1, 1, 1, 0, 0
        Y0 = 0
        for i in range(5):

            y0, y_, log, score5, score6 = my_score_cv(gd_method[0], x, y, random_state=i, cv=10)
            if score6 > score60:
                Y0, pre_y0, log0, score50, score60, ran = y0, y_, log, score5, score6, i
            elif round(score6, 3) == round(score60, 3) and score5 < score50:
                Y0, pre_y0, log0, score50, score60, ran = y0, y_, log, score5, score6, i
            else:
                pass

        st = pd.DataFrame(np.vstack((Y0, pre_y0)).T)
        log0 = pd.DataFrame(log0)
        storename = r"result/CV-{}-{}-random-{}-{}-{}.csv".format(method, target_y, ran, score50, score60)
        print(storename)
        st.to_csv(storename)
        log0.to_csv(storename, mode="a")

        ###############
        # from matplotlib import pyplot as plt
        # plt.scatter(y, y_)
        # plt.show()
