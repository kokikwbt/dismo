""" Main """

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
import tensorly as tl
from termcolor import colored
import dismo
import data.googletrends as gtrends
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='out/tmp')

# Model options

parser.add_argument('--n_interaction', type=int, default=4)
parser.add_argument('--n_seasonality', type=int, default=2)
parser.add_argument('--min_interaction', type=int, default=2)
parser.add_argument('--max_interaction', type=int, default=4)
parser.add_argument('--min_seasonality', type=int, default=2)
parser.add_argument('--max_seasonality', type=int, default=4)
parser.add_argument('--n_season', type=int, default=0)
parser.add_argument('--n_trial', type=int, default=5)
parser.add_argument('--max_time', type=float, default=10)
parser.add_argument('--max_iter', type=int, default=10)
parser.add_argument('--online_max_iter', type=int, default=5)
parser.add_argument('--interaction_type', type=str, default='full')
parser.add_argument('--use_self_interaction', action='store_true')
parser.add_argument('--disuse_carrying_capacity', action='store_false')
parser.add_argument('--non_negative_seasonality', action='store_true')
parser.add_argument('--update_seasonality', action='store_true')
parser.add_argument('--regime_shift', type=int, default=True)
parser.add_argument('--grid_search', type=int, default=False)

# Streaming options

parser.add_argument('--start_point', type=int, default=0)
parser.add_argument('--init_window_size', type=int, default=104)
parser.add_argument('--slide_window_size', type=int, default=52)
parser.add_argument('--report_step', type=int, default=4)
parser.add_argument('--max_forecast_step', type=int, default=4)

# Dataset options

parser.add_argument('--dataset', type=str, default='gtrends')
parser.add_argument('--minmax_scale', action='store_true')

# 1. GoogleTrends

parser.add_argument('--query', type=str, default='vod')
parser.add_argument('--geo_level', type=str, default='region')
parser.add_argument('--sampling_rate', type=str, default='W')
parser.add_argument('--start_date', type=str, default='2008-01-01')
parser.add_argument('--end_date', type=str, default='2021-01-01')

# 2. OnlineRetails

# Output options

parser.add_argument('--track_elapsed_time', action='store_true')
parser.add_argument('--track_grid_search', action='store_true')

args = parser.parse_args()

def make_outputdir(config):

    path = config['outdir']  # root

    if config['dataset'] == 'gtrends':
        path = os.path.join(path,
            config['dataset'],
            config['query'],
            config['geo_level'],
            config['sampling_rate'])

    # elif 

    if config['grid_search'] == True:
        path = os.path.join(path, 'grid_search')
    else:
        path = os.path.join(path,
            'n_interaction={}'.format(config['n_interaction']),
            'n_seasonality={}'.format(config['n_seasonality']))

    if config['regime_shift']:
        path = os.path.join(path, 'regime_shift')
    else:
        path = os.path.join(path, 'no_regime_shift')

    path = os.path.join(path, 'interaction_type='+config['interaction_type'])

    return path

outdir = make_outputdir(vars(args))

if os.path.exists(outdir):
    shutil.rmtree(outdir)

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
# save to a deep directory
args.outdir = outdir
print('output_path=', args.outdir)

if args.dataset == 'gtrends':
    tts = gtrends.load_as_tensor(
        query=args.query,
        geo_level=args.geo_level,
        sampling_rate=args.sampling_rate,
        start_date=args.start_date,
        end_date=args.end_date)

if args.minmax_scale:
    print("MinMaxScale")
    # tts = dismo.utils.MinMaxScaler().fit_transform(tts)
    tts = minmax_scale(tts.reshape((-1, 1))).reshape(tts.shape)

model = dismo.DISMO(
    tts.shape[1:],
    minc=args.min_interaction,
    maxc=args.max_interaction,
    mins=args.min_seasonality,
    maxs=args.max_seasonality,
    n_season=args.n_season,
    n_trial=args.n_trial,
    max_time=args.max_time,
    online_max_iter=args.online_max_iter,
    interaction_type=args.interaction_type,
    use_self_interaction=args.use_self_interaction,
    normalize_ml_projection=False,
    regime_shift=args.regime_shift,
    use_carrying_capacity=args.disuse_carrying_capacity,
    non_negative_seasonality=args.non_negative_seasonality,
    init_complemenatry_matrices=args.update_seasonality)

print(model)

if args.init_window_size > 0:
    init_tensor = tts[args.start_point:args.start_point+args.init_window_size]
else:
    init_tensor = tts

if args.grid_search:
    scores = model.grid_search(
        init_tensor, t=args.start_point, max_iter=args.max_iter)

    # if args.track_grid_search:
    # save results
    dismo.utils.plot_grid_search_result(
        model, scores, args.outdir + '/result_grid_search.png',
        title='Best set= ({}, {})'.format(model.c, model.s))

    model.initialize(init_tensor,
                     t=args.start_point,
                     max_iter=args.max_iter)
else:
    model.initialize(init_tensor,
                     c=args.n_interaction,
                     s=args.n_seasonality,
                     t=args.start_point,
                     max_iter=args.max_iter)


# Experimental settings
config = vars(args)
dismo.utils.saveas_json(args.outdir + "/config.json", config)

ts = args.start_point
dt = args.report_step
wd = args.slide_window_size
ls = args.max_forecast_step
ed = len(tts)

rec_df_list = []
pred_df_list = []
rec_intr_df_list = []
pred_intr_df_list = []
latent_seq_df_list = []
update_time_log = []

for t in range(ts, ed - wd - ls - dt, dt):

    # extract current window
    cur_tensor = tts[t:t+wd]
    print('t=', t, t+wd, cur_tensor.shape)

    # Online update & segmentation
    tic = time.process_time()
    model.update(cur_tensor, t=t)
    toc = time.process_time() - tic
    print(colored('Elapsed time={:.3f} sec.\n'.format(toc), 'blue'))
    update_time_log.append(toc)

    # perform forecasting
    pred, pred_intr, latent_seq, theta = model.fit_predict(ls + dt, cur_tensor, t,
        return_full_sequence=True,
        return_latent_dynamics=True,
        return_dynamics=True,
        return_model=True)

    # save predictions
    rec_, pred = np.split(pred, [len(pred) - (ls + dt)])
    rec_df = dismo.utils.pred2df(rec_, window_index=t)
    pred_df = dismo.utils.pred2df(pred, window_index=t+wd)
    rec_df_list.append(rec_df)
    pred_df_list.append(pred_df)

    try:
        print('RMSE=', np.sqrt(mean_squared_error(
            tts[t+wd+ls:t+wd+ls+dt].ravel(),  # original
            pred[-dt:].ravel()  # predictions
        )))
    except:
        pass

    # save only interactions
    rec_intr, pred_intr = np.split(pred_intr, [len(pred_intr) - (ls + dt)])
    rec_intr_df = dismo.utils.pred2df(rec_intr, window_index=t)
    pred_intr_df = dismo.utils.pred2df(pred_intr, window_index=t+wd)  # window_id: end point of current window
    rec_intr_df_list.append(rec_intr_df)
    pred_intr_df_list.append(pred_intr_df)

    # save latent sequences
    latent_seq_df = dismo.utils.seq2df(latent_seq, window_index=t)
    latent_seq_df_list.append(latent_seq_df)

    # keep overwriting
    # pd.concat(rec_df_list).to_csv(args.outdir + "/rec_.csv.gz", index=False,)
    # pd.concat(pred_df_list).to_csv(args.outdir + "/pred.csv.gz", index=False)
    # pd.concat(rec_intr_df_list).to_csv(args.outdir + '/rec_intr.csv.gz', index=False)
    # pd.concat(pred_intr_df_list).to_csv(args.outdir + '/pred_intr.csv.gz', index=False)
    # pd.concat(latent_seq_df_list).to_csv(args.outdir + '/latent_seq.csv.gz', index=False)

    # Model parameters
    # model.save(args.outdir)

    # break
    # continue
    pd.concat(rec_df_list).to_csv(args.outdir + "/rec_.csv.gz", index=False)
    pd.concat(pred_df_list).to_csv(args.outdir + "/pred.csv.gz", index=False)
    pd.concat(rec_intr_df_list).to_csv(args.outdir + '/rec_intr.csv.gz', index=False)
    pd.concat(pred_intr_df_list).to_csv(args.outdir + '/pred_intr.csv.gz', index=False)
    pd.concat(latent_seq_df_list).to_csv(args.outdir + '/latent_seq.csv.gz', index=False)

# Model parameters
model.save(args.outdir)

# Results
pd.concat(rec_df_list).to_csv(args.outdir + "/rec_.csv.gz")
pd.concat(pred_df_list).to_csv(args.outdir + "/pred.csv.gz")
np.savetxt(args.outdir + '/update_time_log.txt.gz', update_time_log)
