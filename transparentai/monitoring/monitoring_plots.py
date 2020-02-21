import matplotlib.pyplot as plt
import seaborn as sns

import transparentai.utils as utils
import transparentai.plots as plots
        
    
def plot_bar_performance(orig_perf=None, new_perf=None, alert_threshold=None):
    """
    Plot performance in `Monitoring` class with one bar chart for each metric.

    If `orig_perf` is not None but `new_perf` is, then plots only original 
    performance metrics.
    If `orig_perf` is None but `new_perf` is not, then plots only new 
    performance metrics.
    If neither `orig_perf` and `new_perf` are not None, then plots
    comparison between original metrics and new ones.

    You can add a threshold to see if the metrics are good or bad.

    Parameters
    ----------
    orig_perf: dict (default: None)
        Dictionary with metrics in keys and 
        original metric values in values
    new_perf: dict (default: None)
        Dictionary with metrics in keys and 
        new metric values in values 
    alert_threshold: dict (default: None)
        Dictionary with metrics in keys and
        threshold values in values

    Raises
    ------
    AttributeError:
        At least one of `orig_perf` or `new_perf` has to be set.
    """
    if (orig_perf is None) & (new_perf is None):
        raise AttributeError('Either orig_perf or new_perf has to be set')
    
    perf, perf_2 = (orig_perf, new_perf) if orig_perf is not None else (new_perf, None)
    txt, txt_2 = ('Original', 'New') if orig_perf is not None else ('New', '')
    
    height = int((len(perf) + len(perf)%2) / 2)
    fig, ax = plt.subplots(figsize=(15, 4*height))
    labels = []
    
    i = 1
    for k, v in perf.items():
        ax = plt.subplot(int(f'{height}2{i}'))
        plots.plot_bar(ax, value=v, label=k+f' {txt}')
        
        if perf_2 is not None:
            if k in perf_2:
                plots.plot_bar(ax, value=perf_2[k], label=k+f' {txt_2}')
                
        
        if alert_threshold is not None:
            if k in alert_threshold:
                ax.axhline(y=alert_threshold[k], linewidth=2, color='red')
                
        ax.set_title(f'{k} metric', fontsize=16)
        i += 1
        
    
    if alert_threshold is not None:
        labels += ['Alert threshold']
    
    labels += [txt] if perf_2 is None else [txt, txt_2]
    fig.legend(labels=labels, fontsize=16)
    
    fig.suptitle('Performance bar plot', fontsize=18)
    plots.plot_or_save(fname='monitoring_bar_performance_plot')
        
        
def plot_gauge_bias(attr=None, target_value=None, orig_bias=None, new_bias=None):
    """
    Plot bias in `Monitoring` class with one gauge plot for each metric.
    
    If `attr` and `target_value` it reduces the number of plots 
    Example : if you have 2 target values (e.g. 0 and 1) and different protected
    attributes (e.g. gender and marital status) then you can say that it will
    only plots for gender and target value = 1 

    If `orig_bias` is not None but `new_bias` is, then plots only original 
    performance metrics.
    If `orig_bias` is None but `new_bias` is not, then plots only new 
    performance metrics.
    If neither `orig_bias` and `new_bias` are not None, then plots
    comparison between original metrics and new ones.

    Parameters
    ----------
    attr: str (default: None)
        Name of the attribute to analyse
    target_value: (default: None)
        Specific label value 
    orig_bias: dict (default: None)
        Dictionary with metrics in keys and 
        original metric values in values
    new_bias: dict (default: None)
        Dictionary with metrics in keys and 
        new metric values in values 

    Raises
    ------
    AttributeError:
        At least one of `orig_bias` or `new_bias` has to be set.
    """
    if (orig_bias is None) & (new_bias is None):
        raise AttributeError('Either orig_bias or new_bias has to be set')
    
    bias, bias_2 = (orig_bias, new_bias) if orig_bias is not None else (new_bias, None)
    txt, txt_2 = ('Original', 'New') if orig_bias is not None else ('New', '')
    
    colors = ['#34495e','#e67e22']
    
    attributes = [attr] if attr is not None else list(bias.keys())
    
    for attr in attributes:
        n_target = len(bias[attr])
        
        values = [target_value] if target_value is not None else list(bias[attr].keys())        
        for target_value in values:
            metrics = dict(bias[attr][target_value])
            height = int((len(metrics) + len(metrics)%2) / 2)
            fig, ax = plt.subplots(figsize=(15, 2*height))
            
            i = 1
            for metric in metrics:
                ax = plt.subplot(int(f'{height}2{i}'))
                value = metrics[metric]
                goal = utils.get_metric_goal(metric=metric)
                
                plots.plot_gauge(ax, value, goal, bar_color=colors[0])
                        
                if bias_2 is not None:
                    if attr in bias_2:
                        if target_value in bias_2[attr]:                  
                            if metric in bias_2[attr][target_value]:
                                value = bias_2[attr][target_value][metric]
                                ax.axvline(linewidth=4, color=colors[1], x=value)
                
                ax.set_title(f'{metric}')
                i += 1
                
            labels = [txt] if bias_2 is None else [txt, txt_2]
            fig.legend(labels=labels, fontsize=16)
                    
            fig.suptitle(f'Focus on {attr} attribute : bias plot for {target_value} label', fontsize=18)
            plots.plot_or_save(fname=f'monitoring_{attr}_{target_value}_gauge_bias_plot.png')