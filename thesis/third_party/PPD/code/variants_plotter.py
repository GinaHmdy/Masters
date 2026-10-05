import argparse

import matplotlib
import numpy as np
import pandas as pd
import os.path

from matplotlib.lines import Line2D
from sys import platform

from matplotlib.pyplot import ylabel

from constants import column_map, colors_for_plot, all_markers, colors_for_plot_lm, colors_for_plot_dnn, all_markers_lm, \
    all_markers_dnn, colors_for_kl_div, colors_for_layers

if platform == "darwin":
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_init(fsize, xlabel, ylabel):
    plt.figure(figsize=(18, 13.53))
    # plt.figure(figsize=(24, 18)) # accuracy curve
    plt.rc('legend', fontsize=fsize)
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel(xlabel, fontsize=fsize + 8) # 56 for type 2
    plt.ylabel(ylabel, fontsize=fsize + 8)


def generate_line_styles_colors_and_markers(number=12):
    all_markers_cur = all_markers
    linestyles = []
    markers = []
    for i in range(number):
        linestyles.append('-')
        markers.append(all_markers_cur[i % len(all_markers_cur)])
    return linestyles, markers


def draw_phi_vs_ppd(df, type, xlabel='Phi', ylabel='PPD', filename='phi_vs_ppd.pdf', fmt='pdf'):
    """
    Draw a line plot where x-axis = phi, y-axis = PPD.
    """

    plot_init(fsize=48, xlabel=xlabel, ylabel=ylabel)

    if type == 4:
        # Draw line
        plt.plot(df['phi'], df['recall@20'], marker='o', linestyle='-',
                color=colors_for_layers[0], linewidth=5, markersize=10, label='PPD')
    else:
        # Draw line
        plt.plot(df['beta'], df['recall@20'], marker='o', linestyle='-',
                 color=colors_for_layers[0], linewidth=5, markersize=10, label='PPD')

    # Set limits with a small margin
    # axes = plt.gca()
    # min_val, max_val = df['PPD'].min(), df['PPD'].max()
    # axes.set_ylim([min_val - 0.01, max_val + 0.01])

    if type == 4:
        plt.xticks(np.arange(0.0, 1.25, 0.25), fontsize=48)
        plt.yticks(np.arange(.004, .11, .02), fontsize=48)
        # plt.yticks(fontsize=96)
    else:
        plt.xticks(np.arange(0.0, 0.40, 0.10), fontsize=48)
        plt.yticks(np.arange(.083, .100, .004), fontsize=48)
        # plt.yticks(fontsize=96)

    # plt.legend(loc='upper left', ncol=1, fancybox=True, shadow=False)
    # axes.get_legend().remove()

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')


def draw_multi_y_column(df, num_plots, labels, xlabel, ylabel, filename, fmt='pdf', type=2):
    if type == 1 or type == 3:
        colors_for_plot_cur = colors_for_kl_div
    else:
        colors_for_plot_cur = colors_for_plot
    columns = list(df.columns)

    xcol = columns[0]
    ycols = columns[1:]

    plot_init(fsize=48, xlabel=xlabel, ylabel=ylabel)
    legend_handles = []

    linestyles, markers = generate_line_styles_colors_and_markers(num_plots)
    if type == 2:
        # Histogram for PIF vs LightGCN per layer
        # plt.figure(figsize=(12, 6))
        layers = df['layer']

        colors = colors_for_layers

        # Plot PIF
        plt.bar(layers - 0.2, df['PPD'], width=0.4, label='PPD', color=colors[0])
        # Plot LightGCN
        plt.bar(layers + 0.2, df['LightGCN'], width=0.4, label='LightGCN', color=colors[1])
    else:
        ls = 0
        for i in range(num_plots):
            line, = plt.plot(xcol, ycols[i], data=df, linewidth=5, linestyle=linestyles[ls % len(linestyles)],
                             color=colors_for_plot_cur[ls % len(colors_for_plot)], marker=markers[ls % len(markers)], markersize=10)
            legend_handles.append(line)
            ls += 1

    axes = plt.gca()
    min, max = get_max_and_min_values_of_df(df)
    min = min - 0.002
    max = max + 0.002

    axes.set_ylim([min, max])

    if type == 1:
        # axes.set_xlim([4.5, 20.5])
        axes.legend(loc='upper left', bbox_to_anchor=(0, -.1),
                    ncol=5, fancybox=True, shadow=False)
    elif type == 2:
        # axes.set_xlim([0.45, 2.05])
        # axes.set_xlim(0 - 0.07, 4 + 0.07)  # leave space at start & end of x
        axes.legend(loc='upper left', bbox_to_anchor=(0, 1.0),
                    ncol=5, fancybox=True, shadow=False)
    elif type == 3:
        # axes.set_xlim([df['query'].min() - 20, df['query'].max() + 20])
        axes.legend(loc='upper left', bbox_to_anchor=(0, -.1),
                    ncol=5, fancybox=True, shadow=False)
    else:
        # axes.set_xlim([0.05, 21])
        axes.legend(loc='upper left', bbox_to_anchor=(0, -.1),
                    ncol=5, fancybox=True, shadow=False)

    # plt.legend(handles=legend_handles, labels=labels, prop={'size': 32, 'weight': 'bold'}, ncol=3)
    # axes.legend(loc='upper left', bbox_to_anchor=(0, -.1),
    #             ncol=5, fancybox=True, shadow=False)
    if type == 1:
        # axes.legend(loc='upper right', bbox_to_anchor=(0, 0),
                    # ncol=5, fancybox=True, shadow=False)
        axes.legend(loc='upper right', ncol=5, fancybox=True, shadow=False)
        plt.xticks(np.arange(0.0, 5.0, 1.0), fontsize=96)
        # plt.yticks(np.arange(.40, 2.90, .40), fontsize=96)
        # plt.yticks(fontsize=96)
        # plt.xticks(np.arange(5, 21, 5), fontsize=96)
    elif type == 2:
        plt.xticks(np.arange(1.0, 5.0, 1.0), fontsize=96)
        plt.yticks(np.arange(.00, .42, .07), fontsize=96)
        # plt.yticks(fontsize=96)
    elif type == 3:
        axes.legend(loc='lower right', ncol=5, fancybox=True, shadow=False)
        plt.xticks(np.arange(0.0, 5.0, 1.0), fontsize=96)
        plt.yticks(fontsize=96)
        # plt.yticks(np.arange(.75, 0.96, .04), fontsize=96)
    else:
        # tick_positions = df['passes'].values
        # plt.xticks(tick_positions, fontsize=96)
        # tick_labels = ['1', '10', '20']
        # plt.xticks(tick_positions, tick_labels)
        pass
    # plt.yticks(np.arange(.60, .72, .02), fontsize=96)
    # axes.get_legend().remove()

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')


def replace_column_name(result_df):
    new_column_list = {}
    for col in result_df.columns:
        if col in column_map:
            new_column_list[col] = column_map[col]
        else:
            new_column_list[col] = col
    result_df.rename(columns=new_column_list, inplace=True)
    return result_df


def get_max_and_min_values_of_df(df):
    min_max_df = df.agg(['min', 'max'])
    min_val = min((min_max_df.iloc[0:1, 1:]).values.flatten().tolist())
    max_val = max((min_max_df.iloc[1:2, 1:]).values.flatten().tolist())
    return min_val, max_val

def drop_columns_from_results(results):
    # results = results.drop(['naive-svm'], axis=1, errors='ignore')
    # results = results.drop(['pijd_lr'], axis=1, errors='ignore')
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', default='KauiRec_beta_variant.csv', help='result file.')
    parser.add_argument('-out', default='KauiRec_beta_variant.pdf', help='output image filename.')
    parser.add_argument('-fmt', default='pdf', help='image format.')
    parser.add_argument('-metric', default="recall", type=str, help='metric name.')
    parser.add_argument('-type', default=5, type=str, help='type of plotter. 1 for KL, 2 for layers,'
                                                           '3 for number of entropy, 4 phi plotter, 5 beta plotter.')
    parser.add_argument('-p', default=20, help='evaluation position.')
    args = parser.parse_args()

    results = pd.read_csv(args.res)
    results = results.iloc[:, :]
    metric_suffix = args.metric
    p = args.p

    if args.type == 1:
        xlabel = "Popularity bias severity ($s$)"
    elif args.type == 2:
        xlabel = "GNN layer number"
    elif args.type == 3:
        xlabel = "Popularity bias severity ($s$)"
    else:
        xlabel = "Number of passes"

    if args.type == 1:
        ylabel = 'KL divergence'
    elif args.type == 3:
        ylabel = 'Entropy'
    else:
        if metric_suffix == 'recall':
            ylabel = 'Recall@%d' % p
        else:
            ylabel = 'NDCG@%d' % p

    out_file = args.out
    if out_file == '':
        out_file = '%s.%s' % (''.join(args.res.split('.')[:-1]), args.fmt)

    # if 'pvalue' in results.columns:
    #     p_values = results['pvalue']
    # else:
    #     p_values = None

    results = drop_columns_from_results(results)
    results = replace_column_name(results)
    # columns = results.columns
    # results = results.reindex([columns[0],
    #                                columns[3], columns[11], columns[7],
    #                                columns[5], columns[13]],
    #                               axis=1)

    if args.type == 1:
        dropped_colum = 'severity'
    elif args.type == 2:
        dropped_colum = 'layer'
    elif args.type == 3:
        dropped_colum = 'severity'
    elif args.type == 4 or args.type == 5:
        dropped_colum = ''
    else:
        dropped_colum = 'passes'

    if args.type == 4:
        draw_phi_vs_ppd(results, args.type, xlabel=r'Preference centroid coefficient ($\phi$)', ylabel='Recall@20', filename=out_file,
                        fmt=args.fmt)
    elif args.type == 5:
        draw_phi_vs_ppd(results, args.type, xlabel=r'Popularity penalty coefficient ($\beta$)', ylabel='Recall@20',
                        filename=out_file, fmt=args.fmt)
    else:
        draw_multi_y_column(results, results.shape[1] - 1, list(results.columns.drop([dropped_colum])), xlabel, ylabel,
                        out_file, fmt=args.fmt, type=args.type)


if __name__ == "__main__":
    main()