
import pandas
# from check_m1 import preprocessing, pipe_line, measures, kwargs
from check_m2 import preprocessing, pipe_line, measures, kwargs
# from check_m3 import preprocessing, pipe_line, measures, kwargs
# from check_m4 import preprocessing, pipe_line, measures, kwargs


d = './data/dataset_groupgrab.csv'

dataset = pandas.read_csv(d)
kw_preprocess, kw_pipe = kwargs()

n_repeats = 6
n_run = len(kw_preprocess)
step = ((n_run * n_repeats) // 100) + 1
print('Total: {0} steps'.format(n_run * n_repeats))
print('Step Verbosity: {0} steps'.format(step))


trains, vals = [], []
for r in range(n_repeats):
    for k in range(n_run):

        if 48 <= (r * n_run + k) < 1000:

            if (r * n_run + k) % step == 0:
                print(r * n_run + k)

            dataset_pp = preprocessing(data=dataset, kwargs=kw_preprocess[k])

            thresh = int(dataset_pp.shape[0] * 0.8)
            data_train = dataset_pp.iloc[:thresh, :]
            data_val = dataset_pp.iloc[thresh:, :]

            y_train_bench, y_train_hat, y_val_bench, y_val_hat = pipe_line(train=data_train, val=data_val, kwargs=kw_pipe[k])

            # to autotest:
            def do_measure(y_train_bench, y_train_hat, y_val_bench, y_val_hat, measures):
                train_result = {str(measure): measure(y_true=y_train_bench, y_pred=y_train_hat) for measure in measures}
                val_result = {str(measure): measure(y_true=y_val_bench, y_pred=y_val_hat) for measure in measures}
                return train_result, val_result

            train_r, val_r = do_measure(y_train_bench=y_train_bench,
                                        y_train_hat=y_train_hat,
                                        y_val_bench=y_val_bench,
                                        y_val_hat=y_val_hat,
                                        measures=measures())

            train_r, val_r = pandas.Series(train_r), pandas.Series(val_r)
            train_r['k'], val_r['k'] = k, k
            train_r['r'], val_r['r'] = r, r
            trains.append(train_r)
            vals.append(val_r)
trains = pandas.concat(trains, axis=1).T
vals = pandas.concat(vals, axis=1).T
trains = trains.rename(columns={str(x): str(x) + ' TRAIN' for x in trains.columns})
vals = vals.rename(columns={str(x): str(x) + ' VAL' for x in vals.columns})
report = pandas.concat((trains, vals), axis=1)
