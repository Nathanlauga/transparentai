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

| Element (feature) | Questions to answer | feature type | Requirements / information to help |
| ----------------- | ------------------- | ------------ | ---------------------------------- |
| Shareholder participation | Who need this AI ? <br> Who develop it ? <br> Who will use it at the end ? | `meta` | Needs all project participants |
| Fundamental rights | Is this AI will have an impact on people's life in any way ? <br> If yes, scale it. <br> What should be the AI "moral" values ? | `meta` | List of fundamental rights <br> how to define "moral" value | 
| Sustainable, environment friendly | What is the AI's CO² footprint ? <br> How much times did the AI train ? <br> How much data did you use ? | `technical` | Data types ecology impact (theoricaly) <br> compute times ecology impact | 
| Justification | Why do you need this AI (purpose) ? <br> What is your benefit from this AI ? | `meta` | Needs to be validate by a external referent (e.g. DPO) |

# 2. `non-maleficence`


# 3. `autonomy`


# 4. `justice`


# 5. `explicability`