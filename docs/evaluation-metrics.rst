List of preformated metrics
============================

Evaluation metrics
------------------

If you use the compute_metrics function in the :code:`classification.compute_metrics` or :code:`regression.compute_metrics` functions there are preformated metrics.

You can see details of the code in the documentation page : `transparentai.models`_

.. _transparentai.models: https://transparentai.readthedocs.io/en/latest/modules/models.html

This is the list :

+----------------+-----------------------------------+
| Problem type   | metric name                       |
+================+===================================+
| classification | :code:`'accuracy'`                |
+----------------+-----------------------------------+
| classification | :code:`'balanced_accuracy'`       |
+----------------+-----------------------------------+
| classification | :code:`'average_precision'`       |
+----------------+-----------------------------------+
| classification | :code:`'brier_score'`             |
+----------------+-----------------------------------+
| classification | :code:`'f1'`                      |
+----------------+-----------------------------------+
| classification | :code:`'f1_micro'`                |
+----------------+-----------------------------------+
| classification | :code:`'f1_macro'`                |
+----------------+-----------------------------------+
| classification | :code:`'f1_weighted'`             |
+----------------+-----------------------------------+
| classification | :code:`'f1_samples'`              |
+----------------+-----------------------------------+
| classification | :code:`'log_loss'`                |
+----------------+-----------------------------------+
| classification | :code:`'precision'`               |
+----------------+-----------------------------------+
| classification | :code:`'precision_micro'`         |
+----------------+-----------------------------------+
| classification | :code:`'recall'`                  |
+----------------+-----------------------------------+
| classification | :code:`'recall_micro'`            |
+----------------+-----------------------------------+
| classification | :code:`'true_positive_rate'`      |
+----------------+-----------------------------------+
| classification | :code:`'false_positive_rate'`     |
+----------------+-----------------------------------+
| classification | :code:`'jaccard'`                 |
+----------------+-----------------------------------+
| classification | :code:`'matthews_corrcoef'`       |
+----------------+-----------------------------------+
| classification | :code:`'roc_auc'`                 |
+----------------+-----------------------------------+
| classification | :code:`'roc_auc_ovr'`             |
+----------------+-----------------------------------+
| classification | :code:`'roc_auc_ovo'`             |
+----------------+-----------------------------------+
| classification | :code:`'roc_auc_ovr_weighted'`    |
+----------------+-----------------------------------+
| classification | :code:`'roc_auc_ovo_weighted'`    |
+----------------+-----------------------------------+
| classification | :code:`'true_positives'`          |
+----------------+-----------------------------------+
| classification | :code:`'false_positives'`         |
+----------------+-----------------------------------+
| classification | :code:`'false_negatives'`         |
+----------------+-----------------------------------+
| classification | :code:`'true_negatives'`          |
+----------------+-----------------------------------+
| classification | :code:`'confusion_matrix'`        |
+----------------+-----------------------------------+
| regression     | :code:`'max_error'`               |
+----------------+-----------------------------------+
| regression     | :code:`'mean_absolute_error'`     |
+----------------+-----------------------------------+
| regression     | :code:`'mean_squared_error'`      |
+----------------+-----------------------------------+
| regression     | :code:`'root_mean_squared_error'` |
+----------------+-----------------------------------+
| regression     | :code:`'mean_squared_log_error'`  |
+----------------+-----------------------------------+
| regression     | :code:`'median_absolute_error'`   |
+----------------+-----------------------------------+
| regression     | :code:`'r2'`                      |
+----------------+-----------------------------------+
| regression     | :code:`'mean_poisson_deviance'`   |
+----------------+-----------------------------------+
| regression     | :code:`'mean_gamma_deviance'`     |
+----------------+-----------------------------------+
