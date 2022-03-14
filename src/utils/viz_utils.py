import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import ConfusionMatrixDisplay
from itertools import product
from dataset.ExpUtils import STDS, THRESHOLDS

REGR_COLORS = ['#0c2461', '#e58e26']#['yellowgreen', 'mediumseagreen']
RED = '#eb2f06'

def show_training_losses(fn, sb = False, hierarchical = False):
    """
    Plots training losses/errors across training 
    epochs
    """
    # Read metrics
    d = torch.load(fn, map_location='cpu')
    args = d['args']
    print(args)

    mk = 'o'
    total_color = 'dodgerblue'
    color = 'darkorange'
    bin_color = 'rebeccapurple'
    if sb:
        res_color = RED
        regr_colors = REGR_COLORS

    # linestyles
    val_ls = ':'
    train_ls = '-'
    # other_ls = '--'

    # fill styles
    train_fs = 'full'
    val_fs = 'none'

    val_epochs = d['val_epochs']

    # Plot losses
    plt.figure()
    if hierarchical or sb:
        total_losses = d['train_total_losses']
        plt.plot(total_losses, train_ls+mk, label='total loss (training)', color=total_color, fillstyle = train_fs)
    losses = d['train_losses']
    plt.plot(losses, train_ls+mk, label='loss (training)', color=color, fillstyle = train_fs)
    if hierarchical:
        bin_losses = d['train_binary_losses']
        plt.plot(bin_losses, train_ls+mk, label='binary loss (training)', color=bin_color, fillstyle = train_fs)
    if not args['skip_validation']:
        if hierarchical or sb:
            val_total_losses = d['val_total_losses']
            plt.plot(val_total_losses, val_ls+mk, label='total loss (validation)', color=total_color, fillstyle = val_fs)
        val_losses = d['val_losses']
        plt.plot(val_epochs, val_losses, val_ls+mk, label='loss (validation)', color=color, fillstyle = val_fs)
        if hierarchical:
            val_bin_losses = d['val_binary_losses']
            plt.plot(val_epochs, val_bin_losses, val_ls+mk, label='binary loss (validation)', color=bin_color, fillstyle = val_fs)            
    proportion_neg_samples = d['proportion_negative_samples']
    plt.plot(proportion_neg_samples, train_ls+'D', label='proportion of negative samples (training)', color='grey', fillstyle = train_fs)
    plt.xlabel('epoch')
    plt.xticks(range(0,len(losses),5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.minorticks_on()
    plt.grid(b = True, which = 'major')
    plt.grid(b = True, which = 'minor', alpha = 0.4)
    plt.ylim(ymin=-0.05)
    plt.show()

    if sb:
        plt.figure()
        aux_target_names = d['args']['aux_variables']
        regr_losses = list(zip(*d['train_regression_losses']))
        for i in range(len(regr_losses)):
            plt.plot(regr_losses[i], train_ls+mk, label='{} loss (training)'.format(aux_target_names[i]), color=regr_colors[i], fillstyle = train_fs)
        res_penalties = d['train_residual_penalties']
        plt.plot(res_penalties, train_ls+mk, label='residual penalty (training)', color=res_color, fillstyle = train_fs)
        if not args['skip_validation']:
            val_regr_losses = list(zip(*d['val_regression_losses']))
            for i in range(len(val_regr_losses)):
                plt.plot(val_regr_losses[i], val_ls+mk, label='{} loss (validation)'.format(aux_target_names[i]), color=regr_colors[i], fillstyle = val_fs)
            val_res_penalties = d['val_residual_penalties']
            plt.plot(val_res_penalties, val_ls+mk, label='residual penalty (validation)', color=res_color, fillstyle = val_fs)

        plt.xlabel('epoch')
        plt.xticks(range(0,len(losses),5))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.minorticks_on()
        plt.grid(b = True, which = 'major')
        plt.grid(b = True, which = 'minor', alpha = 0.4)
        plt.ylim(ymin=-0.05)
        plt.show()

        pos_mk = '^'
        neg_mk = 'v'
        plt.figure()
        val_regr_error = list(zip(*d['val_regression_error']))
        val_pos_regr_error = list(zip(*d['val_pos_regression_error']))
        val_neg_regr_error = list(zip(*d['val_neg_regression_error']))
        for i in range(len(regr_losses)):
            plt.plot(val_regr_error[i], '--'+mk, label='{} error (validation)'.format(aux_target_names[i]), color=regr_colors[i], fillstyle = 'full')
            plt.plot(val_pos_regr_error[i], val_ls+pos_mk, label='{} error (validation, forest)'.format(aux_target_names[i]), color=regr_colors[i], fillstyle = val_fs)
            plt.plot(val_neg_regr_error[i], val_ls+neg_mk, label='{} error (validation, non-forest)'.format(aux_target_names[i]), color=regr_colors[i], fillstyle = val_fs)
        plt.xlabel('epoch')
        plt.xticks(range(0,len(losses),5))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.minorticks_on()
        plt.grid(b = True, which = 'major')
        plt.grid(b = True, which = 'minor', alpha = 0.4)
        plt.ylim(ymin=-0.05)
        plt.show()
        
def dict_zip(*dicts):
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d[k] for d in dicts if k in d] for k in all_keys}
        
def show_detailed_training_results(fn, sb = False):
    """
    Plots class-specific validation mean/overall metrics stored in dictionary fn across training 
    epochs
    """
    d = torch.load(fn, map_location='cpu')

    # markers
    rec_prec_mk = 'o'
    acc_f1_mk = 'o'

    # fill styles
    mean_fs = 'full'
    overall_fs = 'none'
    precision_fs = 'full'
    recall_fs = 'none'

    # line styles
    mean_ls = '-'
    overall_ls = ':'
    precision_ls = '-'
    recall_ls = ':'

    # line colors for accuracy and f1
    acc_color = 'b'
    f1_color = 'deepskyblue'

    # get metrics
    val_reports = d['val_reports']
    if isinstance(val_reports[0], (tuple, list)):
        val_reports = list(zip(*val_reports))
    elif isinstance(val_reports[0], dict):
        val_reports = dict_zip(*val_reports)
    else:
        val_reports = [val_reports]
    val_epochs = d['val_epochs']
    for vr in val_reports:
        if isinstance(vr, str): #val_reports is a dictionary and not a tuple/list
            print(vr)
            vr = val_reports[vr]
        else:
            print(vr)
        classes = list(vr[0].keys())[:-2]
        print(classes)
        #get class colors
        if len(classes) == 4:
            class_colors = ['gray', 'limegreen', 'darkgreen', 'olive']
        elif len(classes) == 2:
            class_colors = ['gray', 'forestgreen']
        elif len(classes) == 3:
            class_colors = ['limegreen', 'darkgreen', 'olive']
        else:
            np.random.seed(0)
            class_colors = np.random.rand(len(classes), 3)
        # plot recall and precision
        plt.figure()
        for i in range(len(classes)):
            c = classes[i]
            recall = [report[c]['recall'] for report in vr]
            precision = [report[c]['precision'] for report in vr]
            plt.plot(val_epochs, recall, recall_ls+rec_prec_mk, color=class_colors[i], fillstyle = recall_fs, label='recall ' + c)
            plt.plot(val_epochs, precision, precision_ls+rec_prec_mk, color=class_colors[i], fillstyle = precision_fs, label='precision ' + c)
        plt.xlabel('epoch')
        plt.xticks(range(0,val_epochs[-1],5))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.minorticks_on()
        plt.grid(b = True, which = 'major')
        plt.grid(b = True, which = 'minor', alpha = 0.4)
        plt.ylim(ymin=0, ymax=1)
        plt.show()

        # plot accuracy and f-1 score
        
        plt.figure()
        mean_acc = [report['mean']['accuracy'] for report in vr]
        overall_acc = [report['overall']['accuracy'] for report in vr]
        mean_f1 = [report['mean']['f1-score'] for report in vr]
        overall_f1 = [report['overall']['f1-score'] for report in vr]
        plt.plot(val_epochs, mean_acc, mean_ls+acc_f1_mk, color=acc_color, fillstyle = mean_fs, label='mean accuracy')
        plt.plot(val_epochs, overall_acc, overall_ls+acc_f1_mk, color=acc_color, fillstyle = overall_fs, label='overall accuracy')
        plt.plot(val_epochs, mean_f1, mean_ls+acc_f1_mk, color=f1_color, fillstyle = mean_fs, label='mean f1 score')
        plt.plot(val_epochs, overall_f1, overall_ls+acc_f1_mk, color=f1_color, fillstyle = overall_fs, label='overall f1 score')
        plt.xlabel('epoch')
        plt.xticks(range(0,val_epochs[-1],5))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.minorticks_on()
        plt.grid(b = True, which = 'major')
        plt.grid(b = True, which = 'minor', alpha = 0.4)
        plt.ylim(ymin=0, ymax=1)
        plt.show()
    if sb:
        regr_colors = ['yellowgreen', 'mediumseagreen']
        aux_target_names = d['args']['aux_variables']
        val_regr_error = list(zip(*d['val_regression_error']))
        val_pos_regr_error = list(zip(*d['val_pos_regression_error']))
        val_neg_regr_error = list(zip(*d['val_neg_regression_error']))
        # Plot regression losses in a different figure, TEMPORARY (2 variables)
        fig, ax1 = plt.subplots()
        color = regr_colors[0]
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('{} regression error'.format(aux_target_names[0]), color=color)
        ax1.plot(val_epochs, val_regr_error[0], color=color, label = 'all')
        ax1.plot(val_epochs, val_pos_regr_error[0], ':', color=color, label = 'forest')
        ax1.plot(val_epochs, val_neg_regr_error[0], '--', color=color, label = 'non-forest')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(val_epochs, val_regr_error[0], color=color)

        ax2 = ax1.twinx() 
        color = regr_colors[1]
        ax2.set_ylabel('{} regression error'.format(aux_target_names[1]), color=color) 
        ax2.plot(val_epochs, val_regr_error[1], color=color, label = 'all')
        ax2.plot(val_epochs, val_pos_regr_error[1], ':', color=color, label = 'forest')
        ax2.plot(val_epochs, val_neg_regr_error[1], '--', color=color, label = 'non-forest')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.ylim(ymin=0)
        plt.show()

def print_report(report, digits=2):
    """
    Print metrics stored in a dictionary in a readable way.
    Format of the dictionary should follow that of the output of eval_utils.rates2metrics()
    """
    row_names = list(report.keys())
    target_names = row_names[:-2]
    avg_names = row_names[-2:]
    headers = list(report[target_names[0]].keys())
    longest_row_name = max(len(r) for r in row_names) 
    width = max(longest_row_name, digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report_str = head_fmt.format('', *headers, width=width)
    report_str += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 5 + ' {:>9.0f}\n'
    for key in target_names + avg_names:
        row = [key] + list(report[key].values())
        report_str += row_fmt.format(*row, width=width, digits=digits)
        report_str += '\n'
    print(report_str)


def display_norm_cm(cm, class_names = ['NF', 'OF', 'CF'], figsize=(12,3), xlabel="Predicted label", ylabel="Target label"):
    """
    Prints a confusion matrix with 3 different normalizations
        - across columns (precision)
        - across rows (recall)
        - across both axis 
    """
    values_format = '.1f'
    cm = cm.astype(np.float32)
    _, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize)
    #plt.suptitle('Validation results')
    ax1.set_title('Precision (%)')
    ax2.set_title('Recall (%)')
    ax3.set_title('Normalized (%)')
    # Precision
    count = cm.sum(axis=0, keepdims=True)
    count[count == 0] = np.nan
    precision_cm = cm / count * 100
    # cm_disp_0 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(precision_cm, class_names, values_format=values_format, ax=ax1, xlabel=xlabel, ylabel=ylabel)
    # Recall
    count = cm.sum(axis=1, keepdims=True)
    count[count == 0] = np.nan
    recall_cm = cm / count * 100
    # cm_disp_1 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(recall_cm, class_names, values_format=values_format, ax=ax2, xlabel=xlabel, ylabel=ylabel)
    # Fully normalized
    count = cm.sum(keepdims=True)
    count[count == 0] = np.nan
    norm_cm = cm / count * 100
    # cm_disp_2 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(norm_cm, class_names, values_format=values_format, ax=ax3, xlabel=xlabel, ylabel=ylabel)
    
def display_count_cm(cm, class_names = ['NF', 'OF', 'CF'], figsize=(3,2.5), xlabel="Predicted label", ylabel="Target label"):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_cm(cm, class_names, values_format='.0f', ax=ax, xlabel=xlabel, ylabel=ylabel)

def plot_cm(cm, class_names, include_values=True, cmap="Blues", values_format='.1f', ax=None, 
            xticks_rotation='vertical', colorbar=False, xlabel="Predicted label", ylabel="Target label"):
    """
    Adapted from sklearn.metrics.ConfusionMatrixDisplay.plot(), to obtain a fix colormap range
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    CMDisplay = ConfusionMatrixDisplay(confusion_matrix=(cm), display_labels=class_names)
    cm = CMDisplay.confusion_matrix
    n_classes = cm.shape[0]
    # a = 0.1
    # functions = (lambda x: np.log(1+a*x), lambda x: (np.exp(x) - 1)/a )
    CMDisplay.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap, norm=colors.PowerNorm(gamma=0.4, vmin=0, vmax=100))
    CMDisplay.text_ = None
    cmap_min, cmap_max = CMDisplay.im_.cmap(0), CMDisplay.im_.cmap(1.0)

    if include_values:
        CMDisplay.text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = 50 

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], ".2g")
                if cm.dtype.kind != "f":
                    text_d = format(cm[i, j], "d")
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)

            CMDisplay.text_[i, j] = ax.text(
                j, i, text_cm, ha="center", va="center", color=color
            )

    if CMDisplay.display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = CMDisplay.display_labels
    if colorbar:
        fig.colorbar(CMDisplay.im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel=ylabel,
        xlabel=xlabel,
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


def show_model_metrics(fn, epoch=-1, class_names = None, sb = False, s = 0.1, scale = [['log', 'linear'], ['linear']],
                            val_max = [[60, 5], [100]], show_scatter = True, show_2Dhist = False, show_table = True, show_cm = True):
    """
    Prints validation metrics + confusion matrices of the model for a given epoch
    Setting epoch to None indicates that the file contains metrics for one given epoch (as opposed to consecutive 
    training epochs)
    """

    d = torch.load(fn, map_location='cpu')
    if sb:
        figsize = (5, 5)
        aux_target_names = d['args']['aux_variables']
        regr_pred_pts = d['val_regression_prediction_points']
        regr_target_pts = d['val_regression_target_points']
        
        if epoch is not None:
            regr_pred_pts = regr_pred_pts[epoch]
            regr_target_pts = regr_target_pts[epoch]
        if show_2Dhist:
            edges = [[0, 1, 3, 5, 10, 20, 60], np.arange(0, 101, 10)]
            lim = [[1e-1, 60], [0, 100]]
            for i in range(len(aux_target_names)):
                if len(regr_pred_pts[i]) > 0:
                    for scl in scale[i]:
                        plt.figure(figsize=figsize)
                        # plt.hist2d(regr_target_pts[i], regr_pred_pts[i], [edges[i]]*2)
                        # plt.xscale(scale[i])
                        # plt.yscale(scale[i])
                        counts, _, _ = np.histogram2d(regr_target_pts[i], regr_pred_pts[i], bins=(edges[i], edges[i]))
                        plt.pcolormesh(edges[i], edges[i], counts.T, norm=colors.LogNorm(vmin=1, vmax=counts.max()))
                        plt.colorbar()
                        plt.xscale(scl)
                        plt.yscale(scl)
                        plt.xlim(xmin=lim[i][0], xmax=lim[i][1])
                        plt.ylim(ymin=lim[i][0], ymax=lim[i][1])
                        plt.xlabel('Target')
                        plt.ylabel('Prediction')
                        plt.title('{}'.format(aux_target_names[i]))
                        plt.tight_layout()
                        plt.show()
    # Scatter plot for regression tasks
        if show_scatter:
            for i in range(len(aux_target_names)):
                if len(regr_pred_pts[i]) > 0:
                    print('Plotting {} points'.format(len(regr_pred_pts[i])))
                    for scl, ymax in zip(scale[i], val_max[i]):
                        
                        if ymax is None:
                            ymax = np.max(np.concatenate(regr_target_pts[i], regr_pred_pts[i]))
                        mask = (regr_target_pts[i] <= ymax) * (regr_pred_pts[i] <= ymax)
                        print(aux_target_names[i])
                        plt.figure(figsize=figsize)
                        plt.scatter(regr_target_pts[i][mask], regr_pred_pts[i][mask], s=s, c=REGR_COLORS[i], alpha=0.02)#, label="model predictions")
                        id = np.linspace(0, ymax, num=1000)
                        plt.plot(id, id, color='k', label="1:1 line")
                        try:
                            thresholds = d['thresholds'][i]
                        except KeyError:
                            thresholds = THRESHOLDS[aux_target_names[i]]
                        plt.vlines(thresholds, ymin=0, ymax=ymax, colors=[RED]*len(thresholds), linestyle='dashed', label="rule thresholds")
                        plt.hlines(thresholds, xmin=0, xmax=ymax, colors=[RED]*len(thresholds), linestyle='dashed')#, label="thresholds")
                        plt.xlabel('Target')
                        plt.ylabel('Prediction')
                        plt.xscale(scl)
                        plt.yscale(scl)
                        #plt.title('{}'.format(aux_target_names[i]))
                        plt.grid(b=True, which='major')
                        plt.grid(b=True, which='minor', alpha=0.4)
                        plt.legend(bbox_to_anchor=(0.5, 1.1), loc='center')
                        plt.tight_layout()
                        plt.show()

    # Print validation metrics (as a table)
    if show_table or show_cm:
        print('Validation results (epoch {})'.format(epoch))
        report = d['val_reports']
        if epoch is not None:
            report = report[epoch]
        if show_table:
            if isinstance(report, (tuple, list, dict)):
                for r in report:
                    if isinstance(r, str): # cm is a dictionary
                        key = r
                        print(key)
                        r = report[key]
                    print_report(r, digits=2)
            else:
                print_report(report, digits=2)

    # Print confusion matrices
    if show_cm:
        cm = d['val_cms']
        if epoch is not None:
            cm = cm[epoch]
        if isinstance(cm, (tuple, list, dict)):
            for i, m in enumerate(cm):
                if isinstance(m, str): # cm is a dictionary
                    key = m
                    print(key)
                    class_names = list(report[key].keys())[:-2]
                    m = cm[key]
                else:
                    if isinstance(report, (tuple, list)):
                        class_names = list(report[i].keys())[:-2]
                    else:
                        class_names = list(report.keys())[:-2]
                display_norm_cm(m, class_names)
        else:
            display_norm_cm(cm, class_names)


def get_main_metrics(fn, idx = 0):
    """
    Reads loss, mean accury and f-1 scores from an experiment. 
    idx specifies which output to use if the experiment has multiple outputs.
    """
    d = torch.load(fn, map_location='cpu')
    losses = d['train_losses']
    val_reports = d['val_reports']
    val_reports = [report[idx] if isinstance(report, (tuple, list)) else report for report in val_reports]
    mean_acc = [report['mean']['accuracy'] for report in val_reports]
    mean_f1 = [report['mean']['f1-score'] for report in val_reports]
    val_epochs = d['val_epochs']
    return val_epochs, losses, mean_acc, mean_f1

def compare_2_experiments(fn1, fn2, name1, name2):
    """Plots loss, mean accuracy and f-1 score from 2 experiments on the same plot"""

    val_epochs_1, losses_1, mean_acc_1, mean_f1_1 = get_main_metrics(fn1)
    val_epochs_2, losses_2, mean_acc_2, mean_f1_2 = get_main_metrics(fn2)
    marker1 = 'o'
    marker2 = 'x'
    plt.figure()
    plt.plot(losses_1, 'b-' + marker1, label='loss ' + name1)
    plt.plot(losses_2, 'b-' + marker2, label='loss ' + name2)
    plt.plot(val_epochs_1, mean_f1_1, 'g-' + marker1, label='mean f-1 ' + name1)
    plt.plot(val_epochs_2, mean_f1_2, 'g-' + marker2, label='mean f-1 ' + name2)
    #plt.plot(val_epochs_1, mean_acc_1, '-' + marker1, color = 'olive', label='mean accuracy ' + name1)
    #plt.plot(val_epochs_2, mean_acc_2, '-' + marker2, color = 'olive', label='mean accuracy ' + name2)
    plt.xlabel('epoch')
    plt.xticks(range(0,val_epochs_1[-1],5))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.minorticks_on()
    plt.grid(b = True, which = 'major')
    plt.grid(b = True, which = 'minor', alpha = 0.4)
    plt.ylim(ymin=0)
    plt.show()