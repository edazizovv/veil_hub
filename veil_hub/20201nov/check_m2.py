#


#
import numpy
import pandas

from sklearn.metrics import r2_score, mean_squared_error

import torch
from torch import nn

from neuro_kernel import Skeleton


#
def preprocessing(data, kwargs):

    data_ = data.copy()
    data_ = data_.set_index(['Timestamp'])
    data_ = data_.sort_index(ascending=True)

    cols = data_.columns.values

    for j in range(3):
        data_[[x + '_LAG{0}'.format(j) for x in cols]] = data_[cols].shift(periods=j)
    data_ = data_.iloc[2:, :].copy()

    return data_


def pipe_line(train, val, kwargs):

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    train_ = train.copy()
    val_ = val.copy()

    cols = numpy.unique([x[:x.index('_LAG')] if '_LAG' in x else x for x in train_.columns.values])

    for j in range(2):
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

    xx_train = []
    xx_val = []
    yy_train = []
    yy_val = []
    for j in range(x_train.shape[0] - kwargs['window'] + 1):
        xx_train.append(x_train[j:j + kwargs['window'], :].reshape(1, kwargs['window'], x_train.shape[1]))
        yy_train.append(y_train[j:j + kwargs['window']].reshape(1, kwargs['window'], 1))
    for j in range(x_val.shape[0] - kwargs['window'] + 1):
        xx_val.append(x_val[j:j + kwargs['window'], :].reshape(1, kwargs['window'], x_val.shape[1]))
        yy_val.append(y_val[j:j + kwargs['window']].reshape(1, kwargs['window'], 1))
    xx_train_ = numpy.concatenate(xx_train, axis=0)
    xx_val_ = numpy.concatenate(xx_val, axis=0)
    yy_train_ = numpy.concatenate(yy_train, axis=0)
    yy_val_ = numpy.concatenate(yy_val, axis=0)
    xx_train = torch.tensor(xx_train_, dtype=torch.float, device=device)
    xx_val = torch.tensor(xx_val_, dtype=torch.float, device=device)
    yy_train = torch.tensor(yy_train_, dtype=torch.float, device=device)
    yy_val = torch.tensor(yy_val_, dtype=torch.float, device=device)

    model = Skeleton(**kwargs['nn_arch_args'])

    optimizer = kwargs['optimizer']
    optimizer_kwargs = {'lr': kwargs['lr']}
    loss_function = kwargs['loss_function']
    model.fit(xx_train, yy_train, xx_val, yy_val, optimizer, optimizer_kwargs, loss_function, epochs=kwargs['epochs'])

    yy_train_hat = model.predict(x=xx_train)
    yy_val_hat = model.predict(x=xx_val)

    if device.type == 'cuda':
           yy_train_hat = yy_train_hat.cpu()
           yy_val_hat = yy_val_hat.cpu()

    y_train_bench = log_inverse(y_train[kwargs['window'] - 1:].flatten(),
                                train_[target_base].values[kwargs['window'] - 1:])
    y_train_hat = log_inverse(yy_train_hat[:, -1, :].numpy().flatten(),
                              train_[target_base].values[kwargs['window'] - 1:])
    y_val_bench = log_inverse(y_val[kwargs['window'] - 1:].flatten(),
                              val_[target_base].values[kwargs['window'] - 1:])
    y_val_hat = log_inverse(yy_val_hat[:, -1, :].numpy().flatten(),
                            val_[target_base].values[kwargs['window'] - 1:])

    return y_train_bench, y_train_hat, y_val_bench, y_val_hat


def _SMAPE(y_true, y_pred):
    value = (numpy.abs(y_true - y_pred) / ((y_true + y_pred) / 2)).sum() / y_true.shape[0]
    return value


def measures():
    return [r2_score, _SMAPE]


verbose = -1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

kwg = [
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
{'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [10, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [20, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear],
         'layers_dimensions': [100, 1],
         'layers_kwargs': [{}, {}],
         'activators': [None, nn.LeakyReLU],
         'drops': [0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [10, 4, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [20, 8, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear],
         'layers_dimensions': [100, 40, 1],
         'layers_kwargs': [{}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.LSTM, nn.LSTM, nn.LSTM, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, None, None, None, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.1, 0.1, 0.1, 0.1, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.2, 0.2, 0.2, 0.2, 0.0],
         'verbose': verbose,
         'device': device},
        {'layers': [nn.LSTM, nn.Linear, nn.Linear, nn.Linear, nn.Linear],
         'layers_dimensions': [8, 16, 32, 64, 1],
         'layers_kwargs': [{}, {}, {}, {}, {}],
         'activators': [None, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
         'drops': [0.5, 0.5, 0.5, 0.5, 0.0],
         'verbose': verbose,
         'device': device},
    ]

kwargs_pipe = [{'n_lags': 1,
                                                         'nn_arch_args': kwg[k],
                'optimizer': torch.optim.Adam,
'epochs': 10_000,
'loss_function': nn.MSELoss(),
'lr': .001,
'window': 1
                                                         }
               for k in range(len(kwg))] + \
[{'n_lags': 1,
                                                         'nn_arch_args': kwg[k],
                'optimizer': torch.optim.Adam,
'epochs': 10_000,
'loss_function': nn.MSELoss(),
'lr': .001,
'window': 10
                                                         }
               for k in range(len(kwg))] + \
[{'n_lags': 1,
                                                         'nn_arch_args': kwg[k],
                'optimizer': torch.optim.Adam,
'epochs': 10_000,
'loss_function': nn.MSELoss(),
'lr': .001,
'window': 100
                                                         }
               for k in range(len(kwg))]

kwargs_preprocessing = [{'n_lags': 1}] * len(kwg) * 3


def kwargs():
    return kwargs_preprocessing, kwargs_pipe


def log_verse(x, x_lag, eps=1e-11):
    return numpy.log((x + eps) / (x_lag + eps))


def log_inverse(x, x_lag, eps=1e-11):
    return numpy.multiply((numpy.exp(x) - eps), (x_lag - eps))

