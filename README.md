# TransparentAI
*Is my AI ethic ?*


TransparentAI is a toolbox in Python to answer the question "Is my AI ethic ?" based on the European Commission requierements.

[![Documentation](https://readthedocs.org/projects/transparentai/badge/?version=latest)](http://transparentai.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/transparentai.svg)](https://badge.fury.io/py/transparentai)

## Why this tool ?

The research of ethic in the Artificial Intelligence field is a hot topic. More than 70 papers between 2016 and 2019 ([Algorithm Watch, 2019](https://algorithmwatch.org/en/project/ai-ethics-guidelines-global-inventory/)). But many papers just present the question of "What should be an ethic AI" not the "How to do it". In consequence, many developers become fustrated and still don't really know to do it in practive ([Peters, May 2019](https://medium.com/ethics-of-digital-experience/beyond-principles-a-process-for-responsible-tech-aefc921f7317)).

`TransparentAI` is an answer to this question. The philosophy is that, in coherence with the [Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation) by the European Commission, you can find out easily (in Python) if **"your AI is ethic"** !

**New tool :** This is a new tool so if you found any bugs or other kind of problems please do not hesitate to report them on the
issues GitHub page from the library here : https://github.com/Nathanlauga/transparentai/issues. I hope you will enjoy this tool and help me to improve it!

Documentation is available here : [API Documentation](https://transparentai.readthedocs.io/en/latest/).

## Table of content

1. [Installation](#installation)
2. [Compatible model and data type](#)
3. [Getting started](#)
    - [Is my model biased ?](#)
    - [How can I explain my model ?](#)
    - [What's my model performance ?](#)
    - [What is in my data ?](#)
    - [How can I know is still good over time ?](#)
    - [Is my model sustainable ?](#)
    - [Do I use safe packages ?](#)
4. [UE Commision requirements](#)
5. [Contributing](#)
6. [Credits and ressources](#)
7. [Author](#)
8. [License](#)

## Installation

You can install it with [PyPI](https://pypi.org/project/transparentai/) :
```
pip install transparentai
```

Or by cloning [GitHub repository](https://github.com/Nathanlauga/transparentai/)

```
git clone https://github.com/Nathanlauga/transparentai.git
cd transparentai
python setup.py install
```

*****

## Compatible model and data type

**Version 0.2** :

- Data : can only handle tabular dataset.
- Model : can only handle classification and regression model.

*****

## Getting started 

Take a look on the [Getting started](https://transparentai.readthedocs.io/en/latest/getting-started.html) page of the documenation or you can search specific use cases in the [`examples/`](examples/) directory.

*****

## UE Commision requirements

The European Commission defined seven requirements that allow to make a trustworthy AI.

These requirements are applicable to different stakeholders partaking in AI systems’ life cycle: developers, deployers and end-users, as well as the broader society. By developers, we refer to those who research, design and/or develop AI systems. By deployers, we refer to public or private organisations that use AI systems within their business processes and to offer products and services to others. End-users are those engaging with the AI system, directly or indirectly. Finally, the broader society encompasses all others that are directly or indirectly affected by AI systems. Different groups of stakeholders have different roles to play in ensuring that the requirements are met:

- Developers should implement and apply the requirements to design and development processes.
- Deployers should ensure that the systems they use and the products and services they offer meet the requirements.
- End-users and the broader society should be informed about these requirements and able to request that they are upheld.

The below list of requirements is non-exhaustive. 35 It includes systemic, individual and societal aspects:

1. **Human agency and oversight**:
Including fundamental rights, human agency and human oversight
2. **Technical robustness and safety**:
Including resilience to attack and security, fall back plan and general safety, accuracy, reliability and reproducibility
3. **Privacy and data governance**:
Including respect for privacy, quality and integrity of data, and access to data
4. **Transparency**:
Including traceability, explainability and communication
5. **Diversity, non-discrimination and fairness**:
Including the avoidance of unfair bias, accessibility and universal design, and stakeholder participation
6. **Societal and environmental wellbeing**: 
Including sustainability and environmental friendliness, social impact, society and democracy
7. **Accountability**: 
Including auditability, minimisation and reporting of negative impact, trade-offs and redress.

<div style='width=100%;text-align:center;'>
<img src='images/en_7_requirements.png' width='440'>
</div>

This table allows you to in details for each requirements and if it's possible how to control if the aspect is ethic with `TransparentAI`. Some aspects do not have technical implementation in this tool because it requires legal or other knowledge. If you want to understand the differents aspect and requirements you can read details in the [Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation) paper.

| UE requirements | Aspect | `TransparentAI` implementation |
| ----------- | ------------- | ----------- |
| **1. Human agency and oversight** | Fundamental rights | No technical implementation. |
|                            | Human agency | No technical implementation. |
|                            | Human oversight | [Control AI performance over time with `monitoring.monitor_model` or `monitoring.plot_monitoring`]() |
| **2. Technical robustness and safety** | Resilience to attack and security | [Try different input scenario in the model to see how it handles it with `models.explainers.ModelExplainer`]() |
|                            | Fallback plan and general safety | [Check if your Python's package are secure with `utils.check_packages_security`]() |
|                            | Accuracy | [Validate your AI performance with `models.classification.plot_performance` or `models.regression.plot_performance`]() |
|                            | Reliability and Reproducibility | No technical implementation. |
| **3. Privacy and data governance** | Privacy and data protection | No technical implementation. |
|                            | Quality and integrity of data | [Check if the variable is coherent in its distribution with `datasets.variable.plot_variable`]() |
|                            | Access to data | No technical implementation. |
| **4. Transparency** | Traceability | [Generate a performance validation report with `utils.reports.generate_validation_report`]() |
|                            | Explainability | [Explain the local or global behavior of your model with `models.explainers.ModelExplainer`]()|
|                            | Communication | No technical implementation. |
| **5. Diversity, non-discrimination and fairness** | Avoidance of unfair bias |[Check if your AI is biased on protected attributes with `fairness.model_bias` or `fairness.plot_bias`]() |
|                            | Accessibility and universal design | No technical implementation. |
|                            | Stakeholder Participation | No technical implementation. |
| **6. Societal and environmental well-being** | Sustainable and environmentally friendly AI | [Get the kWh value of the AI training with `utils.evaluate_kWh`]() |
|                            | Social impact | [Check if your AI is biased on protected attributes with `fairness.model_bias` or `fairness.plot_bias`]() |
| **7. Accountability** | Auditability | [Generate a performance validation report with `utils.reports.generate_validation_report`]() |
|                            | Minimisation and reporting of negative impacts | No technical implementation. |
|                            | Trade-offs | No technical implementation. |
|                            | Redress | No technical implementation. |

*****

## Contributing

See the [contributing file](CONTRIBUTING.md).

*PRs accepted.*

*****

## Credits and ressources

### `fairness` submodule

For this submodule, I have to say I was mainly inspired by one tool so all the credit has to be attributed to [AIF360 by IBM](http://aif360.mybluemix.net/).

I used some of the metrics proposed in the tools (`Statistical Parity Difference`, `Equal Opportunity Difference`, `Average Odds Difference`, `Disparact Impact` and `Theil Index`).

### `models.evaluation` submodule

I used some metrics functin of the [`scikit-learn`](https://scikit-learn.org/stable/index.html) Python package.

### `models.explainers` submodule

I choose to used the [Shap](https://github.com/slundberg/shap) library because this tool was tested and aproved by a lot of people in the community, and even if I found some papers showing some problems (e.g. "Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods" [(Slack and al., November 2019)](https://arxiv.org/abs/1911.02508)), I decided to use it because if you want to biased Shap, you have to do it intentionally at the AI creation.

### Plotting functions

I was inspired by some graphics on [Kaggle](https://www.kaggle.com/). But mainly I use some code on [matplotlib website](https://matplotlib.org/) and the [Python graph gallery](https://python-graph-gallery.com/).

### `utils.external` functions

I used different packages that implement great features such as :

1. [`energyusage`](https://github.com/responsibleproblemsolving/energy-usage) : A Python package that measures the environmental impact of computation.
2. [`safety`](https://github.com/pyupio/safety) : Safety checks your installed dependencies for known security vulnerabilities.

Again thanks to researchers and developers that contributed in this really important field, without them I don't think I'll be able to create this tool.

*****

## Author

This work is led by [Nathan Lauga](https://github.com/nathanlauga/), french Data Scientist.

*****

## License

This project use a [MIT License](LICENSE).

**Why ?**

I believe that the code should be re-used for community projects and also inside private projects. 
AI transparency needs to be available for everyone even it's a private AI! 