#


#
import numpy
import pandas

from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX


#
def preprocessing(data, kwargs):

    data_ = data.copy()
    data_ = data_.set_index(['Timestamp'])
    data_ = data_.sort_index(ascending=True)

    data_[[x + '_LAG1' for x in data_.columns.values]] = data_[data_.columns.values].shift(periods=1)
    data_ = data_.iloc[1:, :].copy()

    return data_


def pipe_line(train, val, kwargs):

    train_ = train.copy()
    val_ = val.copy()

    for col in [x for x in train_.columns.values if 'LAG' not in x]:
        train_[col] = log_verse(train_[col].values, train_[col + '_LAG1'].values)
        val_[col] = log_verse(val_[col].values, val_[col + '_LAG1'].values)
    train_ = train_.iloc[1:, :].copy()
    val_ = val_.iloc[1:, :].copy()

    if kwargs['model'] == 'SARIMAX':
        target = 'BTC-USD_Close||quantile_0.5'
        target_base = 'BTC-USD_Close||quantile_0.5_LAG1'

        y_train = train_[target].values
        y_val = val_[target].values

        model = SARIMAX(endog=y_train, order=kwargs['order'], trend=kwargs['trend'])
        model = model.fit(method=kwargs['method'])

        yy_train_hat = model.predict().reshape(-1, 1)
        yy_val_hat = model.forecast(steps=y_val.shape[0]).reshape(-1, 1)

        y_train_bench = log_inverse(y_train.flatten(),
                                    train_[target_base].values)
        y_train_hat = log_inverse(yy_train_hat.flatten(),
                                  train_[target_base].values)
        y_val_bench = log_inverse(y_val.flatten(),
                                  val_[target_base].values)
        y_val_hat = log_inverse(yy_val_hat.flatten(),
                                val_[target_base].values)
    elif kwargs['model'] == 'VARMAX':
        target = 'BTC-USD_Close||quantile_0.5'
        target_base = 'BTC-USD_Close||quantile_0.5_LAG1'
        targets = [x for x in train_.columns.values if 'LAG' not in x and target not in x] + [target]

        y_train = train_[targets].values
        y_val = val_[targets].values

        model = VARMAX(endog=y_train, order=kwargs['order'], trend=kwargs['trend'])
        model = model.fit(method=kwargs['method'])

        yy_train_hat = model.predict()[:, -1].reshape(-1, 1)
        yy_val_hat = model.forecast(steps=y_val.shape[0])[:, -1].reshape(-1, 1)

        y_train_bench = log_inverse(y_train[:, -1],
                                    train_[target_base].values)
        y_train_hat = log_inverse(yy_train_hat.flatten(),
                                  train_[target_base].values)
        y_val_bench = log_inverse(y_val[:, -1],
                                  val_[target_base].values)
        y_val_hat = log_inverse(yy_val_hat.flatten(),
                                val_[target_base].values)
    else:
        raise Exception("you've passed a wrong model mate")

    return y_train_bench, y_train_hat, y_val_bench, y_val_hat


def _SMAPE(y_true, y_pred):
    value = (numpy.abs(y_true - y_pred) / ((y_true + y_pred) / 2)).sum() / y_true.shape[0]
    return value


def measures():
    return [r2_score, _SMAPE]


sm_orders = [(j, 0,  0) for j in [1, 16, 70, 81, 93, 99]] + \
         [(j, 0,  1) for j in [1, 16, 70, 81, 93, 99]] + \
         [(j, 0, 16) for j in [1, 16, 70, 81, 93, 99]]

vm_orders = [(1, 0, 0), (1, 0, 1), (1, 0, 16)]
trend = 'n'
method = 'powell'

kwargs_pipe = [{'model': 'SARIMAX',
                                                         'method': method,
'order': sm_orders[k],
'trend': trend,
                                                         }
               for k in range(len(sm_orders))] + \
[{'model': 'VARMAX',
                                                         'method': method,
'order': vm_orders[k],
'trend': trend,
                                                         }
               for k in range(len(vm_orders))]

kwargs_preprocessing = [{}] * (len(sm_orders) + len(vm_orders))


def kwargs():
    return kwargs_preprocessing, kwargs_pipe


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))

