column_map = {
    # 'dml_m53_lm': 'CFC',
    # 'dml_m54_lm': 'CFC (KDE)',
    # 'dml_m56_lm': 'CFC (IMR)',
    # 'dml_m57_lm': 'CFC (PDF)',
    # 'dml_m59_lm': 'CFC (PMF)',
    'dml_m51_lm': 'CFC',
    'dml_m53_lm': 'CFC',
    'dml_m57_lm': 'CFC',
    'dml_lm': 'CFC (LambdaMART)',
    'dml_dnn': 'CFC (DNN)',

    'dml_m53_svmr': 'CFC-RankSVM',
    'dml_m54_svmr': 'CFC-RankSVM (KDE)',
    'dml_m56_svmr': 'CFC-RankSVM (IMR)',
    'dml_m57_svmr': 'CFC-RankSVM (PDF)',
    'dml_m59_svmr': 'CFC-RankSVM (PMF)',

    'naive-svm' : 'RankSVM',
    'ml_lm' : 'LambdaMART',
    'dnn' : 'DNN',
    'true_lm': 'Oracle (LambdaMART)',
    'true_dnn': 'Oracle (DNN)',

    'dla' : 'DLA (DNN)',
    'paird' : 'PairD (LambdaMART)',
    'rem' : 'REM (DNN)',
    'multr' : 'MULTR (DNN)',
    'upe' : 'UPE (DNN)',
    'pal': 'PAL (DNN)',
    'pijd_lm' : 'PIJD (LambdaMART)',
    'pijd_dnn' : 'PIJD (DNN)',
    'pijd_lr' : 'PIJD-LR',
}

# colors_for_plot = ['red', 'peru', 'orchid', 'darkblue', 'green', 'orange', 'purple', 'cyan', 'olive', 'skyblue',
                   # 'darkgoldenrod', 'brown', 'peru', 'violet', 'thistle', 'salmon' 'blue']
# colors_for_plot = ['red', 'brown', 'peru', 'orchid', 'darkblue', 'green', 'tan', 'orange', 'purple', 'cyan',
                   # 'salmon', 'olive']
colors_for_plot = ['red', 'darkblue', 'green', 'cyan', 'olive',
                   'brown', 'steelblue', 'peru', 'orchid', 'tan', 'orange', 'purple', 'salmon', 'pink']
colors_for_plot_lm = ['red', 'darkblue', 'green', 'cyan', 'olive']
colors_for_plot_dnn = ['brown', 'steelblue', 'peru', 'orchid', 'tan', 'orange', 'purple', 'salmon', 'pink']
significance_color = 'darkgreen'
all_markers = ["p", "^", "v", '<', '>',
               'd', 'D', 's', "o", 'x', '.', "+", '*', 'h']
all_markers_lm = ["p", "^", "v", '<', '>']
all_markers_dnn = ['d', 'D', 's', "o", 'x', '.', "+", '*', 'h']

colors_for_kl_div = ['purple', 'salmon']

colors_for_layers = ['#1f77b4', '#ff7f0e']