European Commission 7 requirements for a Trustworthy AI
------

Link to the [Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation) by the European Commission.

This table allows you to in details for each requirements and if it's possible how to control if the aspect is ethic with `TransparentAI`. 

| UE requirements | Aspect | `TransparentAI` implemenation |
| ----------- | ------------- | ----------- |
| 1. Human agency and oversight | [Fundamental rights](requirements/1-Fundamental_rights.md) | can't be : need legal knowledge |
|                            | [Human agency]() | Declarative answer |
|                            | [Human oversight]() | [Control AI performance over time with `monitoring.monitor_model` or `monitoring.plot_monitoring`]() |
| 2. Technical robustness and safety | [Resilience to attack and security]() | [Try different input scenario in the model to see how it handles it with `models.explainers.ModelExplainer`]() |
|                            | [Fallback plan and general safety]() | [Check if your Python's package are secure with `utils.check_packages_security`]() |
|                            | [Accuracy]() | [Validate your AI performance with `models.classification.plot_performance` or `models.regression.plot_performance`]() |
|                            | [Reliability and Reproducibility]() | |
| 3. Privacy and data governance | [Privacy and data protection]() | can't be : need DPO approval |
|                            | [Quality and integrity of data]() | [Check if the variable is coherent in its distribution with `datasets.variable.plot_variable`]() |
|                            | [Access to data]() | |
| 4. Transparency | [Traceability]() | [Generate a performance validation report with `utils.reports.generate_validation_report`]() |
|                            | [Explainability]() | [Explain the local or global behavior of your model with `models.explainers.ModelExplainer`]()|
|                            | [Communication]() | |
| 5. Diversity, non-discrimination and fairness | [Avoidance of unfair bias]() |[Check if your AI is biased on protected attributes with `fairness.model_bias` or `fairness.plot_bias`]() |
|                            | [Accessibility and universal design]() | |
|                            | [Stakeholder Participation]() | Declarative answer |
| 6. Societal and environmental well-being | [Sustainable and environmentally friendly AI]() | [Get the kWh value of the AI training with `utils.evaluate_kWh`]() |
|                            | [Social impact]() | [Check if your AI is biased on protected attributes with `fairness.model_bias` or `fairness.plot_bias`]() |
| Accountability | [Auditability]() | [Generate a performance validation report with `utils.reports.generate_validation_report`]() |
|                            | [Minimisation and reporting of negative impacts]() | Declarative answer |
|                            | [Trade-offs]() | Declarative answer |
|                            | [Redress]() | Declarative answer |
