from utils import plt

import pandas as pd
import seaborn as sns
import re

def parse_log(model, fname):
    data = []

    if 'hybrid' in model or 'gp' in model:
        uncertainty = 'GP-based uncertainty'
    elif model == 'mlper5g' or model == 'bayesnn':
        uncertainty = 'Other uncertainty'
    else:
        uncertainty = 'No uncertainty'

    with open(fname) as f:
        for line in f:

            starts_with_year = re.search(r"^20[0-9][0-9]-", line)
            if starts_with_year is None:
                continue
            if ' | ' not in line or ' for ' not in line:
                continue

            log_body = line.split(' | ')[1]

            [ metric, log_body ] = log_body.split(' for ')

            [ quadrant, log_body ] = log_body.split(': ')

            if metric == 'MAE':
                continue
            elif metric == 'MSE':
                value = float(log_body)
            elif metric == 'Pearson rho':
                value = float(log_body.strip('()').split(',')[0])
            elif metric == 'Spearman r':
                value = float(log_body.split('=')[1].split(',')[0])
            else:
                continue

            data.append([ model, metric, quadrant, value, uncertainty ])

    return data

def parse_log_dgraphdta(model, fname):
    data = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('metrics for  davis_full'):
                if line.endswith('davis_full'):
                    quadrant = 'unknown_all'
                elif line.endswith('(quadA)'):
                    quadrant = 'unknown_side'
                elif line.endswith('(quadB)'):
                    quadrant = 'unknown_repurpose'
                elif line.endswith('(quadC)'):
                    quadrant = 'unknown_novel'
                else:
                    raise ValueError('Invalid line {}'.format(line))

                value = float(f.readline().rstrip().split()[-1])
                data.append([ model, 'MSE', quadrant, value,
                              'No uncertainty' ])

                value = float(f.readline().rstrip().split()[-1])
                data.append([ model, 'Pearson rho', quadrant, value,
                              'No uncertainty' ])

                value = float(f.readline().rstrip().split()[-1])
                data.append([ model, 'Spearman r', quadrant, value,
                              'No uncertainty' ])

    return data

if __name__ == '__main__':

    models = [
        'mlper1',
        'mlper1norm',
        'mlper1normsklearn',
        'mlper1split',
        'ridgesplit',
        'ridgesplit_morgan',
        'gpsplit',
        'hybrid'

        #1gp',
        #'hybrid',
        #'bayesnn',
        #'mlper5g',
        #'mlper1',
        #'cmf',
        #'dgraphdta'
    ]

    data = []
    for model in models:
        if model == 'dgraphdta':
            for seed in range(5):
                fname = ('../DGraphDTA/iterate_davis2011kinase_dgraphdta_'
                         'seed{}.log'.format(seed))
                data += parse_log_dgraphdta(model, fname)
        else:
            fname = 'target/log/train_davis2011kinase_{}.log'.format(model)
            data += parse_log(model, fname)


    df = pd.DataFrame(data, columns=[
        'model', 'metric', 'quadrant', 'value', 'uncertainty',
    ])

    quadrants = sorted(set(df.quadrant))
    metrics = sorted(set(df.metric))

    for quadrant in quadrants:
        for metric in metrics:

            df_subset = df[(df.metric == metric) &
                           (df.quadrant == quadrant)]

            plt.figure(figsize=(20,10))
            sns.barplot(x='model', y='value', data=df_subset, ci=None,
                        order=models, hue='uncertainty', dodge=False,
                        palette=sns.color_palette("Reds_r", 
                                                  n_colors=len(models)))
            sns.swarmplot(x='model', y='value', data=df_subset, color='black',
                          order=models)
            if (metric == 'Pearson rho' or metric == 'Spearman r') \
               and quadrant != 'unknown_all':
                plt.ylim([ -0.05, 0.7 ])
            if metric == 'MSE' and quadrant != 'unknown_all':
                plt.ylim([ -0.01e7, 3e7 ])
            #plt.savefig('figures/benchmark_cv_{}_{}.svg'
            #            .format(metric, quadrant))
            plt.savefig('figures/benchmark_cv_{}_{}.png'
                        .format(metric, quadrant))
            plt.close()
