import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import ConfusionMatrixDisplay
from itertools import product

REGR_COLORS = ['#0c2461', '#e58e26']
RED = '#eb2f06'

def show_training_losses(fn, hierarchical = False):
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
    if hierarchical:
        total_losses = d['train_total_losses']
        plt.plot(total_losses, train_ls+mk, label='total loss (training)', color=total_color, fillstyle = train_fs)
    losses = d['train_losses']
    plt.plot(losses, train_ls+mk, label='loss (training)', color=color, fillstyle = train_fs)
    if hierarchical:
        bin_losses = d['train_binary_losses']
        plt.plot(bin_losses, train_ls+mk, label='binary loss (training)', color=bin_color, fillstyle = train_fs)
    if not args['skip_validation']:
        if hierarchical:
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
        
def dict_zip(*dicts):
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d[k] for d in dicts if k in d] for k in all_keys}
        
def show_detailed_training_results(fn):
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


def display_cm(cm, class_names = ['NF', 'OF', 'CF']):
    """
    Prints a confusion matrix with 3 different normalizations
        - across columns (precision)
        - across rows (recall)
        - across both axis 
    """
    cm = cm.astype(np.float32)
    _, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3))
    #plt.suptitle('Validation results')
    ax1.set_title('Precision (%)')
    ax2.set_title('Recall (%)')
    ax3.set_title('Normalized (%)')
    # Precision
    count = cm.sum(axis=0, keepdims=True)
    count[count == 0] = np.nan
    precision_cm = cm / count * 100
    # cm_disp_0 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(precision_cm, class_names, values_format ='.1f', ax=ax1, xticks_rotation = 'vertical')
    # Recall
    count = cm.sum(axis=1, keepdims=True)
    count[count == 0] = np.nan
    recall_cm = cm / count * 100
    # cm_disp_1 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(recall_cm, class_names, values_format ='.1f', ax=ax2, xticks_rotation = 'vertical')
    # Fully normalized
    count = cm.sum(keepdims=True)
    count[count == 0] = np.nan
    norm_cm = cm / count * 100
    # cm_disp_2 = ConfusionMatrixDisplay(confusion_matrix=(), display_labels=class_names)
    plot_cm(norm_cm, class_names, values_format ='.1f', ax=ax3, xticks_rotation = 'vertical')

def plot_cm(cm, class_names, include_values=True, cmap="Blues", values_format = '.1f', ax = None, 
            xticks_rotation = 'vertical', colorbar=False):
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
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


def show_model_metrics(fn, epoch=-1, class_names = None, show_table = True, show_cm = True):
    """
    Prints validation metrics + confusion matrices of the model for a given epoch
    Setting epoch to None indicates that the file contains metrics for one given epoch (as opposed to consecutive 
    training epochs)
    """

    d = torch.load(fn, map_location='cpu')

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
                display_cm(m, class_names)
        else:
            display_cm(cm, class_names)


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