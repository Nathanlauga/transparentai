TransparentAI v0.2 : is my AI ethical ?
-----

Based on the work of [Morley and al, September 2019](https://arxiv.org/pdf/1905.06876.pdf), I rethink the TransparentAI approch and then I decided for v0.2 to not think this module logic from the Machine Learning pipeline but from the ethical principle.

So the question is : **How can I know if an AI is ethical ?** 

Let's first remind us of the 5 ethical principles describes by [Floridi & Cowls, June 2019](https://hdsr.mitpress.mit.edu/pub/l0jsh9d1) :

| Principle       | What is it ?                                                           |
| --------------- | ---------------------------------------------------------------------- |
| Beneficence     | Must be benefical to humanity, sustainable development and well-being  | 
| Non-maleficence | must not infringe on privacy or undermine security (Robust and safety) | 
| Autonomy        | Must be human centered                                                 |
| Justice         | Must promote prosperity and fairness                                   |
| Explicability   | Must be understandable and explainable                                 |

Currently I have 6 sub-modules in TransparentAI : 
- `datasets` : analyse datasets variables and composition (only structured)
- `explainer` : explain predictions and model comportment (only linear and tree)
- `fairness` : compute bias on dataset and model
- `models` : analyse model performance (only classification and regression)
- `monitoring` : compare two sample to see if performance decrease or not
- `start` : ask questions to find out if the AI is viable or not

Even if I'm proud of this code, I think it's far from being perfect. First of all some features are not really understandable. Also, some bricks are missing.

*****
So now, let's rethink TransparentAI based on the **5 principles of ethics**, see what I can keep, how should I restructured my code and what feature to implement.
*****

I want to think my code that way :
```bash
├── transparentai
    ├── beneficence
    ├── nonmaleficence
    ├── autonomy
    ├── justice
    ├── explicability

```

Some features needs to be technical but some can't be technical so for the next part I define the following terms :
- `technical features` : feature that be computed technically (It's not just a question answer)
- `meta features` : feature regarding the AI details, you can understand it just by answering a question


# 1. `beneficence`

Elements table (feature to implement) :

| Element (feature) | Questions to answer | feature type | Requirements / information to help | keyword |
| ----------------- | ------------------- | ------------ | ---------------------------------- | ------- |
| Shareholder participation | Who need this AI ? <br> Who develop it ? <br> Who will use it at the end ? | `meta` | Needs all project participants | `people`, `frame` |
| Fundamental rights | Is this AI will have an impact on people's life in any way ? <br> If yes, scale it. <br> What should be the AI "moral" values ? | `meta` | List of fundamental rights <br> how to define "moral" value | `people`, `frame` |
| Sustainable, environment friendly | What is the AI's CO² footprint ? <br> How much times did the AI train ? <br> How much data did you use ? | `technical` | Data types ecology impact (theoricaly) <br> compute times ecology impact | `data`, `model`, `ecology` |
| Justification | Why do you need this AI (purpose) ? <br> What is your benefit from this AI ? | `meta` | Needs to be validate by a external referent (e.g. DPO) | `people`, `frame`, `validation` |

# 2. `non-maleficence`


| Element (feature) | Questions to answer | feature type | Requirements / information to help | keyword |
| ----------------- | ------------------- | ------------ | ---------------------------------- | ------- |
| Resilience to attacks | Does the AI is robbust to attacks ? <br> What did I try to t est the AI security ? | `technical` | Robustness package <br> Checklist to test AI safety | `data`, `model`, `security` |
| Fallback plan | Do I have a backup plan if something goes wrong ? | `meta` | Define a processus if there is a problem | `security` |
| General safety | What technology does the AI use ? <br> Did I check if the technology is safe ? | `meta` `technical` | Search online of security breach if the technology <br> Implement basic problems of main AI technologies | `security` |
| Accuracy & performance | Is the model working with good performance ? <br> Did I try my model on a test set ? <br> If the model is working in the real world, what are the perf ? | `technical` | use `models` module | `model`, `validation` |
| Privacy & data protection |
| Reliability & Reproducibility |
| Quality & integrity | Is there any mistakes in the data ? <br> What rules can ensure me of the data quality ? | `technical` | [Best practices for creating data quality rules](https://blog.syncsort.com/2017/10/big-data/best-practices-data-quality-rules/) | `data` |
| Social impact | What social attribute can be find in the data ? <br> Does the model favour a particular attribute ? | `technical` | Use `fairness` module <br> list of social attribute | `analyse`, `data` | 


# 3. `autonomy`

| Element (feature) | Questions to answer | feature type | Requirements / information to help | keyword |
| ----------------- | ------------------- | ------------ | ---------------------------------- | ------- |
| Decisions | Can the human make decision instead of the AI ? <br> Does the human understand how the model gives a prediction ? | `technical` | module `explainer` <br> get insight in natural language | `model`, `validation` |
| Oversight | Is the model doing good in real case, after deployment ? <br> If the model is a re-entering one, does the prediction exclude new possible scenario ? | `technical` | use `monitoring` module | `monitoring` |

# 4. `justice`

| Element (feature) | Questions to answer | feature type | Requirements / information to help | keyword |
| ----------------- | ------------------- | ------------ | ---------------------------------- | ------- |
| Avoidance of unfair bias | If some bias are detected, is it understandable given AI's goal ? <br> Should I remove the bias or just mitigate them ? | `technical` | use `fairness` module <br> use `2 - Social impact` and `1 - Fundamental rights` with a define threshold to validate or not | `fairness`, `data`, `model` |
| Universal design |
| Auditability | Is the model development documented ? <br> Same for model paramater choose ? <br> Same for data pipeline ? | `meta` `technical` | Documentation is indepedent <br> Auto generatation of a pipeline | `model`, `data`, `documentation` |
| Minimisation & reporting of negative impact | How can you say the AI is doing something wrong ? <br> Define a threshold for bias or/and performance | `meta` `technical` | Define threshold <br> Mitigate bias | `validation`, `monitoring` |
| Trade-off & redress |
 
# 5. `explicability`

| Element (feature) | Questions to answer | feature type | Requirements / information to help | keyword |
| ----------------- | ------------------- | ------------ | ---------------------------------- | ------- |
| Tracability | Is the data used documented ? <br> Is the model pipeline documented ? | `meta` `technical` | Same as `4 - Auditability` | `model`, `data`, `documentation` | 
| Explainability | How does the model is working in general ? <br> Same for one prediction ? | `technical` | use `explainer` module <br> get insight in natural language and graphics | `model`, `validation` |
| Interpretability | Is the model mathematically complex ? | `technical` | Map model type with complexity | `model` | 