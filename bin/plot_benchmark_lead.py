from utils import *

import pandas as pd
import seaborn as sns
import re

def parse_log(model, fname):
    data = []
    reseed = -1
    lead_num = 0

    if model == 'gp' or 'hybrid' in model:
        uncertainty = 'GP-based uncertainty'
    elif model == 'mlper5g' or model == 'bayesnn':
        uncertainty = 'Other uncertainty'
    else:
        uncertainty = 'No uncertainty'

    with open(fname) as f:

        while True:
            line = f.readline()
            if not line:
                break

            starts_with_year = re.search(r"^20[0-9][0-9]-", line)
            if starts_with_year is None:
                continue
            if not ' | ' in line:
                continue

            line = line.split(' | ')[1]

            if line.startswith('Iteration'):
                lead_num = 0
                reseed += 1
                continue

            elif line.startswith('\tAcquire '):
                fields = line.strip().split()

                Kd = 10000 - float(fields[-1])
                chem_idx = int(fields[1].lstrip('(').rstrip(','))
                prot_idx = int(fields[2].strip().rstrip(')'))
                chem_name = fields[3]
                prot_name = fields[4]

                data.append([
                    model, Kd, lead_num, reseed, uncertainty,
                    chem_name, prot_name, chem_idx, prot_idx
                ])

                lead_num += 1
                continue

    return data

def parse_log_dgraphdta(model, fname, seed):
    data = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip()
            if line == 'metrics for  davis_full':
                f.readline()
                f.readline()
                f.readline()

                Kds = [ float(Kd) for Kd in
                        f.readline().rstrip().split(', ') ]
                for lead_num, Kd in enumerate(Kds):
                    data.append([
                        model, Kd, lead_num, seed, 'No uncertainty',
                        None, None, None, None
                    ])

    return data

if __name__ == '__main__':
    models = [
        'hybrid',
        #'hybrid',
        #'bayesnn',
        #'mlper5g',
        'mlper1',
        'ridgesplit'
        #'cmf',
        #'dgraphdta'
    ]

    data = []
    for model in models:
        for seed in range(5):
            if model == 'dgraphdta':
                    fname = ('../DGraphDTA/iterate_davis2011kinase_dgraphdta_'
                             'seed{}.log'.format(seed))
                    data += parse_log_dgraphdta(model, fname, seed)
            else:
                fname = f'iterate/log/iterate_davis2011kinase_{model}_{seed}.log'
                if os.path.exists(fname): 
                    data += parse_log(model, fname)

    df = pd.DataFrame(data, columns=[
        'model', 'Kd', 'lead_num', 'seed', 'uncertainty',
        'chem_name', 'prot_name', 'chem_idx', 'prot_idx',
    ])

    n_leads = [ 5, 25 ]

    for n_lead in n_leads:
        df_subset = df[df.lead_num < n_lead]

        plt.figure()
        sns.barplot(x='model', y='Kd', data=df_subset, ci=95,
                    order=models, hue='uncertainty', dodge=False,
                    palette=sns.color_palette("RdBu", n_colors=len(models),),
                    capsize=0.2,)
        plt.ylim([ -100, 10100 ])
        #plt.savefig('figures/benchmark_lead_{}.svg'.format(n_lead))
        plt.savefig('figures/benchmark_lead_{}.png'.format(n_lead))
        plt.close()

        gp_Kds = df_subset[df_subset.model == 'gp'].Kd
        for model in models:
            if model == 'hybrid':
                continue
            other_Kds = df_subset[df_subset.model == model].Kd
            print('{} leads, t-test, GP vs {}:'.format(n_lead, model))
            print('\tt = {:.4f}, P = {:.4g}'
                  .format(*ss.ttest_ind(gp_Kds, other_Kds,
                                        equal_var=False)))
        print('')
