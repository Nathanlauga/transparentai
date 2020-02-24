# Ressources used for this library

First of all, I need to say that this library was created for my thesis which I will present to a jury in September 2020.

## Why this tool ?

I started to read and search about Ethic and AI since November 2018. I found out about some great tools and research paper, but something hit me : the fact that a lot of paper **(more than 70)** are published between 2016 and 2019 [(Algorithm Watch, 2019)](https://algorithmwatch.org/en/project/ai-ethics-guidelines-global-inventory/) but those papers stays theorical and don't go to the "how".

So as consequences, many tech makers are frustrated by how little help they provide in actual practice. Principles must be sufficiently abstract to retain truth across contexts, but this abstraction also leaves them too vague to be useful for specific design decisions on their own [(Peters, May 2019)](https://medium.com/ethics-of-digital-experience/beyond-principles-a-process-for-responsible-tech-aefc921f7317).

And after reading the paper *"From What to How: An Initial Review of Publicly Available AI Ethics Tools, Methods and Research to Translate Principles into Practices"* [(Morley, Floridi, Kinsey and Elhalal, September 2019)](https://arxiv.org/abs/1905.06876) which do an overview of all the principles described in the differents theorical papers, I decided to use my knowledge to create a tool from the start to the monitoring of the AI **(the current version is just a beginning)**.

I hope you will enjoy this tool and help me to improve it!

*****

## Technicals ressources and inspiration

### `Start` submodule

1. [ML Cannvas](https://www.louisdorard.com/machine-learning-canvas) : a canvas that provide a great overlook about different aspect of the AI. 
2. [PAIR Guidebook](https://pair.withgoogle.com/) : PAIR advances the research and design of people-centric AI systems. They are interested in the full spectrum of human interaction with AI, from supporting the engineers and teams building AI to understanding people’s everyday experiences with AI.

### `datasets` submodule

The plotting function that plots each variable and different variable combinations was inspired by the [AutoViz](https://github.com/AutoViML/AutoViz) Python's library which can visualize any dataset, any size with a single line of code.

### `fairness` submodule

For this submodule, I have to say I was mainly inspired by one tool so all the credit has to be attributed to [AIF360 by IBM](http://aif360.mybluemix.net/).

I used some of the metrics proposed in the tools (`Statistical Parity Difference`, `Equal Opportunity Difference`, `Average Odds Difference`, `Disparact Impact` and `Theil Index`).

### `explainer` submodule

I choose to used the [Shap](https://github.com/slundberg/shap) library because this tool was tested and aproved by a lot of people in the community, and even if I found some papers showing some problems (e.g. "Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods" [(Slack and al., November 2019)](https://arxiv.org/abs/1911.02508)), I decided to use it because if you want to biased Shap, you have to do it intentionally at the AI creation.

### Plotting functions

I was inspired by some graphics on [Kaggle](https://www.kaggle.com/). But mainly I use some code on [matplotlib website](https://matplotlib.org/) and the [Python graph gallery](https://python-graph-gallery.com/).

*****

Again thanks to researchers and developers that contributed in this really important field, without them I don't think I'll be able to create this tool.

*Thanks*,

Nathan.