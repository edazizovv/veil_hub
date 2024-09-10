#


#
import numpy
import pandas

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


#
def preprocessing(data, kwargs):

    data_ = data.copy()
    data_ = data_.set_index(['Timestamp'])
    data_ = data_.sort_index(ascending=True)

    cols = data_.columns.values

    for j in range(kwargs['n_lags'] + 2):
        data_[[x + '_LAG{0}'.format(j) for x in cols]] = data_[cols].shift(periods=j)
    data_ = data_.iloc[kwargs['n_lags'] + 1:, :].copy()

    return data_


def pipe_line(train, val, kwargs):

    train_ = train.copy()
    val_ = val.copy()

    cols = numpy.unique([x[:x.index('_LAG')] if '_LAG' in x else x for x in train_.columns.values])

    for j in range(kwargs['n_lags'] + 1):
        for col in cols:
            train_[col + '_LAG{0}'.format(j) + '_lnpct'] = log_verse(train_[col + '_LAG{0}'.format(j)].values, train_[col + '_LAG{0}'.format(j + 1)].values)
            val_[col + '_LAG{0}'.format(j) + '_lnpct'] = log_verse(val_[col + '_LAG{0}'.format(j)].values, val_[col + '_LAG{0}'.format(j + 1)].values)

    target = 'BTC-USD_Close||quantile_0.5_LAG0_lnpct'
    target_base = 'BTC-USD_Close||quantile_0.5_LAG1'
    factors = [x for x in train_.columns.values if 'LAG0' not in x and '_lnpct' in x]

    x_train = train_[factors].values
    y_train = train_[target].values
    x_val = val_[factors].values
    y_val = val_[target].values

    model = make_pipeline(*kwargs['model_args'])
    model.fit(X=x_train, y=y_train)

    yy_train_hat = model.predict(X=x_train)
    yy_val_hat = model.predict(X=x_val)

    y_train_bench = log_inverse(y_train.flatten(),
                                train_[target_base].values)
    y_train_hat = log_inverse(yy_train_hat.flatten(),
                              train_[target_base].values)
    y_val_bench = log_inverse(y_val.flatten(),
                              val_[target_base].values)
    y_val_hat = log_inverse(yy_val_hat.flatten(),
                            val_[target_base].values)

    return y_train_bench, y_train_hat, y_val_bench, y_val_hat


def _SMAPE(y_true, y_pred):
    value = (numpy.abs(y_true - y_pred) / ((y_true + y_pred) / 2)).sum() / y_true.shape[0]
    return value


def measures():
    return [r2_score, _SMAPE]



kw_stumped_lea_sta = {'n_estimators': 1000, 'max_depth': 2, 'learning_rate': 0.3, 'subsample': 0.8, 'n_jobs': -1}
kw_pruned_lea_sta = {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.3, 'subsample': 0.8, 'n_jobs': -1}
kw_dense_lea_sta = {'n_estimators': 1000, 'max_depth': 1000, 'learning_rate': 0.3, 'subsample': 0.8, 'n_jobs': -1}
kw_stumped_loa_sta = {'n_estimators': 1000, 'max_depth': 2, 'learning_rate': 0.03, 'subsample': 0.8, 'n_jobs': -1}
kw_pruned_loa_sta = {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.8, 'n_jobs': -1}
kw_dense_loa_sta = {'n_estimators': 1000, 'max_depth': 1000, 'learning_rate': 0.03, 'subsample': 0.8, 'n_jobs': -1}


kwg_100 = [
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (XGBRegressor(**kw_stumped_lea_sta),),
       (XGBRegressor(**kw_pruned_lea_sta),),
       (XGBRegressor(**kw_dense_lea_sta),),
       (XGBRegressor(**kw_stumped_loa_sta),),
       (XGBRegressor(**kw_pruned_loa_sta),),
       (XGBRegressor(**kw_dense_loa_sta),)
       ]

kwg_10 = [
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(117 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(117 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (XGBRegressor(**kw_stumped_lea_sta),),
       (XGBRegressor(**kw_pruned_lea_sta),),
       (XGBRegressor(**kw_dense_lea_sta),),
       (XGBRegressor(**kw_stumped_loa_sta),),
       (XGBRegressor(**kw_pruned_loa_sta),),
       (XGBRegressor(**kw_dense_loa_sta),)
       ]


kwg_1 = [
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_stumped_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_stumped_lea_sta)),
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_pruned_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_pruned_lea_sta)),
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_dense_lea_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_dense_lea_sta)),
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_stumped_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_stumped_loa_sta)),
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_pruned_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_pruned_loa_sta)),
       (PCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (PCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.1), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.2), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.4), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.6), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (KernelPCA(n_components=int(35 * 0.8), kernel='sigmoid'), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.1)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.2)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.4)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.6)), XGBRegressor(**kw_dense_loa_sta)),
       (SparsePCA(n_components=int(35 * 0.8)), XGBRegressor(**kw_dense_loa_sta)),
       (XGBRegressor(**kw_stumped_lea_sta),),
       (XGBRegressor(**kw_pruned_lea_sta),),
       (XGBRegressor(**kw_dense_lea_sta),),
       (XGBRegressor(**kw_stumped_loa_sta),),
       (XGBRegressor(**kw_pruned_loa_sta),),
       (XGBRegressor(**kw_dense_loa_sta),)
       ]


kwargs_pipe = [{'n_lags': 100,
                                                         'model_args': kwg_100[k],
                                                         }
               for k in range(len(kwg_100))] + \
[{'n_lags': 10,
                                                         'model_args': kwg_10[k],
                                                         }
               for k in range(len(kwg_10))] + \
[{'n_lags': 1,
                                                         'model_args': kwg_1[k],
                                                         }
               for k in range(len(kwg_1))]

kwargs_preprocessing = [{'n_lags': 100}] * len(kwg_100) + [{'n_lags': 10}] * len(kwg_10) + [{'n_lags': 1}] * len(kwg_1)


def kwargs():
    return kwargs_preprocessing, kwargs_pipe


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))

