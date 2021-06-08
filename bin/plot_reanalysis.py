#!/usr/bin/env python
# coding: utf-8

# # make_figs.ipynb
# *Author:* Sam Goldman
#
# This file code to produce plots from log files contained in "iterate/log" and "target/log"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import re
import copy
import json
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.patches as mpatches
import re

from sklearn import metrics as sklearn_metrics

sns.set(context="paper",
        style="white",
        font_scale=3.5,
        palette="Blues_r",
        rc={
            "figure.figsize": (20, 10),
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "lines.linewidth": 3,
            "font.family": "sans-serif",
            "font.sans-serif": "Arial"
        })

# Matplotlib export
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Global constants

method_name_map = {
    "mlper1": "MLP",
    "hybrid": "GP + MLP",
    "ridgesplit": "Ridge: pretrained",
    "ridgesplit_morgan": "Ridge: Morgan",
    "hybridsplit": "GP + MLP",
    "mlper1split": "MLP",
}


# Helpers
def get_colors(x: list):
    """ Get colors for x, a list of lists, where each sublist gets its own hue"""
    num_cats = len(x)

    color_list = sns.color_palette("Paired", n_colors=num_cats * 2)
    colors = []
    for index, i in enumerate(x):

        num_new_cols = len(i)

        start_col = color_list[index * 2]
        end_col = color_list[index * 2 + 1]
        start, end = np.array(start_col), np.array(end_col)
        new_cols = np.linspace(start, end, num_new_cols).tolist()
        colors.extend(new_cols)
    return colors


def parse_log_exploit(model, fname):
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
                    model, Kd, lead_num, reseed, uncertainty, chem_name,
                    prot_name, chem_idx, prot_idx
                ])

                lead_num += 1
                continue

    return data


def parse_log_cv(model, fname):
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

            [metric, log_body] = log_body.split(' for ')

            [quadrant, log_body] = log_body.split(': ')

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

            data.append([model, metric, quadrant, value, uncertainty])

    return data


def make_cv_plots():
    """Make cv plots """
    models = [
        'mlper1',
        'hybrid',
        'ridgesplit',
        'ridgesplit_morgan',
        'hybridsplit',
        'mlper1split',
    ]

    panel_height = 3.5
    panel_width = 5.5

    orig_models = ["hybrid", "mlper1"]
    no_cpi = ["hybridsplit", "mlper1split"]
    ours = ["ridgesplit", "ridgesplit_morgan"]

    uq_methods = ["hybrid", "hybridsplit"]
    joint_list = [orig_models, no_cpi, ours]

    full_list = [x for i in joint_list for x in i]
    gap_size = 0.3
    shift_factors = np.array(
        [index * gap_size for index, i in enumerate(joint_list) for j in i])
    plot_positions = np.arange(len(full_list)) + shift_factors

    x_labels = ["Hie et al.\n(CPI)"] * len(orig_models) + [
        "Hie et al.\n(No CPI)"
    ] * len(no_cpi) + ["Linear\n(No CPI)"] * len(ours)

    x_colors = get_colors(joint_list)
    color_map = dict(zip(full_list, x_colors))  # dict of colors

    out_dir = "results/figures/cv"
    os.makedirs(out_dir, exist_ok=True)

    # Get data frame
    data = []
    for model in models:
        for seed in range(5):
            fname = f'target/log/train_davis2011kinase_{model}_{seed}.log'
            if os.path.exists(fname):
                data += parse_log_cv(model, fname)
            else:
                print(fname)

    df = pd.DataFrame(data,
                      columns=[
                          'model',
                          'metric',
                          'quadrant',
                          'value',
                          'uncertainty',
                      ])

    metric_map = {
        "Pearson rho": r"Pearson $\rho$",
        "Spearman r": r"Spearman $\rho$",
    }

    quadrants = np.array([
        'observed', 'unknown_all', 'unknown_side', 'unknown_repurpose',
        'unknown_novel'
    ])
    metrics = np.array(['MSE', 'Pearson rho', 'Spearman r'])

    for quadrant in quadrants[1:]:
        quadrant_df = df[df['quadrant'] == quadrant]

        for metric in metrics:
            metric_df = quadrant_df[quadrant_df['metric'] == metric]
            plt.figure(figsize=(panel_width, panel_height))
            bars = []
            for color, plot_position, method in zip(x_colors, plot_positions,
                                                    full_list):
                temp_df = metric_df[metric_df["model"] == method]
                vals = temp_df["value"]
                bar_height = np.mean(vals)
                error_height = 1.96 * stats.sem(vals)
                label_name = method_name_map[method]
                hatch = "/" if method in uq_methods else None
                hatch = None
                bars.append(
                    plt.bar(plot_position,
                            bar_height,
                            color=color,
                            label=label_name,
                            width=1.01,
                            hatch=hatch))
                plt.errorbar(plot_position,
                             bar_height,
                             yerr=error_height,
                             color="Black",
                             capsize=5,
                             capthick=2,
                             linewidth=1)

            # Set labels
            ticks, labels = [], []
            for label in np.unique(x_labels):
                # Cnter on middle box
                avg_pos = np.mean(
                    np.array(plot_positions)[np.array(x_labels) == label])
                ticks.append(avg_pos)
                labels.append(label)
    #         for tick, label in zip(ticks, labels):
    #             plt.text(tick, -0.4, label, rotation=30,
    #                     ha="center", va="center")#label, (tick, -0.2), )

            if (metric == 'Pearson rho'
                    or metric == 'Spearman r') and quadrant != 'unknown_all':
                plt.ylim([0.0, 0.8])
            if metric == 'MSE' and quadrant != 'unknown_all':
                plt.ylim([-0.01e7, 3e7])
            plt.xticks(ticks,
                       labels=labels,
                       rotation=50,
                       horizontalalignment="center")
            plt.ylabel(metric_map.get(metric, metric))

            handles = []
            #         handles.append(mpatches.Patch(facecolor="black", alpha=0.9,hatch="///",label="Uncertainty"))
            #         handles.append(mpatches.Patch(facecolor="black", alpha=0.9,hatch="",label="No Uncertainty"))
            bars.extend(handles)
            plt.legend(
                handles=bars,
                ncol=int(len(full_list) / 2),  # + 1), 
                bbox_to_anchor=(0.5, -1.5),
                loc="lower center",
            )

            # Right legend
            #         plt.legend(handles=bars, ncol=1,#int(len(full_list) / 2),# + 1),
            #                bbox_to_anchor = (1.5, 0.5),loc="center", )

            save_name = os.path.join(out_dir,
                                     f'benchmark_cv_{metric}_{quadrant}.pdf')
            plt.savefig(save_name, bbox_inches="tight")
            print(save_name)
            plt.close()
    return color_map


def make_exploit_plots(color_map):
    """ Make exploit plots """
    out_dir = "results/figures/exploit"
    os.makedirs(out_dir, exist_ok=True)

    models = ['hybrid', 'mlper1', 'ridgesplit']

    model_renames = {
        "hybrid": "Hie et al: GP+MLP",
        "mlper1": "Hie et al: MLP ",
        "ridgesplit": "Linear (No CPI): Ridge"
    }

    panel_height = 3.5
    panel_width = 5.5
    panel_gap = np.array([0, 0, 1]) * 0.6

    plot_positions = np.arange(len(models)) + panel_gap

    #ticks = plot_positions
    #tick_labels = [model_renames[model] for model in models]

    ticks = [np.mean(plot_positions[:1]), plot_positions[2]]
    tick_labels = ["Hie et al.\n(CPI)", "Ridge\n(No CPI)"]

    data = []
    for model in models:
        for seed in range(5):
            fname = f'iterate/log/iterate_davis2011kinase_{model}_{seed}.log'
            if os.path.exists(fname):
                data += parse_log_exploit(model, fname)

    df = pd.DataFrame(data,
                      columns=[
                          'model',
                          'Kd',
                          'lead_num',
                          'seed',
                          'uncertainty',
                          'chem_name',
                          'prot_name',
                          'chem_idx',
                          'prot_idx',
                      ])

    df['LogKd'] = np.log10(df['Kd'])
    df['Kdmicro'] = (df['Kd']) / 1000  # convert fromm nM to micromolar

    x_label_map = {
        "LogKd": "$Log(K_d)$",
        "Kd": "$K_d$",
        "Kdmicro": "True $K_d$ ($\mu M$)"
    }
    val_name = "Kdmicro"

    n_leads = [5, 25]

    for n_lead in n_leads:
        df_subset = df[df.lead_num < n_lead]
        plt.figure(figsize=(panel_width, panel_height))
        bars = []
        for plot_position, method in zip(plot_positions, models):
            temp_df = df_subset[df_subset["model"] == method]

            color = color_map[method]
            vals = temp_df[val_name]
            bar_height = np.mean(vals)
            error_height = 1.96 * stats.sem(vals)
            label_name = method_name_map[method]

            uq = pd.unique(temp_df['uncertainty'])
            if len(uq) > 1:
                raise ValueError()
            uq = uq[0]

            #hatch = "/" if not uq == "No uncertainty" else None
            hatch = None

            bars.append(
                plt.bar(plot_position,
                        bar_height,
                        color=color,
                        label=label_name,
                        width=1.01,
                        hatch=hatch))
            plt.errorbar(plot_position,
                         bar_height,
                         yerr=error_height,
                         color="Black",
                         capsize=5,
                         capthick=2,
                         linewidth=1)

        plt.xticks(ticks,
                   labels=tick_labels,
                   rotation=0,
                   horizontalalignment="center")
        plt.ylabel(x_label_map[val_name])
        #     plt.xlabel("Method")
        #     plt.yscale("log")
        save_name = os.path.join(out_dir, f'benchmark_lead_{n_lead}.pdf')
        print(save_name)
        #     plt.ylim([0.025, 40])
        #     plt.yticks([0.1,1,10], labels= ["$10^{-1}$", "$10^{0}$", "$10^{1}$"])
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    color_map = make_cv_plots()
    make_exploit_plots(color_map)
