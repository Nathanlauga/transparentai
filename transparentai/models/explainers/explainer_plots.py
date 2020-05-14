import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transparentai import plots

def plot_global_feature_influence(feat_importance, color='#3498db', **kwargs):
    """
    Display global feature influence sorted.
    
    Parameters
    ----------
    feat_importance: pd.Series
        Feature importance with feature as indexes and 
        shap value as values
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(feat_importance.values.shape) == 1:
        ax.barh(feat_importance.index, feat_importance.values, 
                label=feat_importance.index, align='center',
                color=color)
    else:
        index = np.arange(len(feat_importance.index))
        bar_width = (1 / len(feat_importance.columns))
        bar_width *= 0.8
        
        for i, v in feat_importance.iteritems():
            ax.barh(index+bar_width*i, v.values, bar_width,
                    label='%ith class'%i, align='center',
                    color=color[i])
        
        plt.yticks(index + bar_width, feat_importance.index)
        plt.legend()
        
    plt.title('Feature importance (using Shap)')
    plt.ylabel('Features')
    plt.xlabel('Global value influence')
    
    # plt.show()   
    return plots.plot_or_figure(fig, **kwargs)
           
        
def plot_local_feature_influence(feat_importance, base_value, pred, pred_class=None, **kwargs):
    """
    Display local feature influence sorted for a specific
    prediction.
    
    Parameters
    ----------
    feat_importance: pd.Series
        Feature importance with feature as indexes and 
        shap value as values
    base_value: number
        prediction value if we don't put any feature into the model
    pred: number
        predicted value
    """    
    feat_importance = feat_importance.sort_values(ascending=False)
    
    current_val = base_value
    left = list()
    for feat, value in feat_importance.items():
        left.append(current_val)
        current_val += value
    left = list(reversed(left))
        
    feat_importance = feat_importance.sort_values()
        
    fig, ax = plt.subplots(figsize=(12, 8))
    line_based = ax.axvline(x=base_value, linewidth=2, color='black', linestyle='--')
    line_pred = ax.axvline(x=pred, linewidth=2, color='black')
    
    pos_color, neg_color = '#2ecc71', '#e74c3c'
    colors = [pos_color if v > 0 else neg_color for v in feat_importance.values]
    
    rects = ax.barh(feat_importance.index, 
                    feat_importance.values, 
                    label=feat_importance.index, 
                    color=colors,
                    left=left,
                    align='center')    
    
    for r in rects:
        h, w, x, y = r.get_height(), r.get_width(), r.get_x(), r.get_y()
        if h == 0:
            continue
        val = round(w,4)
        x = x+w if val > 0 else x
        ax.annotate(str(val), xy=(x, y+h/2), xytext=(0, 0),
                    textcoords="offset points", va='center',
                    color='black', clip_on=True)
    
    xlim0, xlim1 = ax.get_xlim()
    size = xlim1-xlim0
    ax.set_xlim(xlim0-size*0.001, xlim1+size*0.1)
    title = 'Feature importance (using Shap)' 
    title = title if pred_class is None else title + ' - Model output : %s'%str(pred_class)
    plt.title(title)
    plt.ylabel('Features')
    plt.xlabel('Local value influence')
    
    green_patch = mpatches.Patch(color=pos_color)
    red_patch = mpatches.Patch(color=neg_color)
    ax.legend([line_based, line_pred, green_patch, red_patch], 
              ['Based value', 'Prediction', 'Positive influence', 'Negative influence'])
    
    # plt.show()
    return plots.plot_or_figure(fig, **kwargs)
