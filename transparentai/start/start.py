from IPython.display import display, Markdown
import json
import os

import transparentai.utils as utils


def print_md(text):
    """
    """
    return display(Markdown(text))


def get_questions():
    """
    """
    fpath = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../src/json/'))

    with utils.OpenFile(fpath+'/questions.json', mode='r') as file:
        questions = json.load(file)
    file.close()
    return questions


def how_can_i_start(md=True):
    """
    """
    print_fun = print if not md else print_md

    title = 'How can I start' if not md else '### How can I start'
    print_fun(title)

    print_fun('This function is a helper to show some starting possibilities')
    print_fun('')

    print_fun('├── `transparentai.start.quick_start()` shows you questions about the project. ' +
              'If you complete it, at the end you will have answered questions about if your AI is viable.')
    print_fun('├── `transparentai.start.external_link()` shows you external references that can be more accurate ' +
              'to your AI.')


def quick_start(md=True, answer=None, with_input=True):
    """
    """
    print_fun = print if not md else print_md
    questions = get_questions()
    answered_question = questions if answer is None else answer

    title = 'Quick start' if not md else '### Quick start'
    print_fun(title)

    canvas = 'ML Canvas' if not md else '[ML Canvas](https://www.louisdorard.com/machine-learning-canvas)'
    pair = 'PAIR Guidebook' if not md else '[PAIR Guidebook](https://pair.withgoogle.com/)'
    print_fun(f'Those questions were inspired by {canvas} and {pair}.')

    print_fun('')

    for section, sect_questions in questions.items():
        title = section if not md else f'#### {section}'
        print_fun(title)

        for k, v in sect_questions.items():
            q = v['question']
            print_fun(f'`{k}` : {q}')
            if with_input:
                ans = input('Answer here : ')
                answered_question[section][k]['answer'] = ans

    return answered_question


def format_answer(answer, format='json'):
    """

    Accepted format json, txt and md
    """
    ans_return = '' if format != 'json' else {}

    questions = get_questions()

    for section, sect_questions in questions.items():
        for k, v in sect_questions.items():
            q = v['question']
            a = v['answer']
            if format == 'json':
                ans_return[k] = {
                    'question': q,
                    'answer': a
                }
            else:
                ans_return += f"{q}\nAnswer: {a}\n\n"

    return ans_return


def save_answer_to_file(answer, fname, format='json'):
    """
    """
    answer = format_answer(answer, format=format)
    if format != 'json':
        utils.str_to_file(string=answer, fpath=fname)
    else:
        with utils.OpenFile(fname, mode='w') as file:
            json.dump(answer, file)
        file.close()


def format_link(url, md_text, md):
    """
    """
    return url if not md else f'[{md_text}]({url})'


def external_link(md=True):
    """
    """
    print_fun = print if not md else print_md

    title = 'External links' if not md else '### External links'
    print_fun(title)
    print_fun(
        'This is the list of different links that can be usefull to start an AI (with more illustrations)')

    print_fun('')

    url = 'https://deon.drivendata.org/'
    md_text = 'deon'
    url = format_link(url, md_text, md=md)
    print_fun(f'├── {url} – Deon by DrivenData is a command line tool that allows you ' +
              'to easily add an ethics checklist to your data science projects.')

    url = 'https://www.louisdorard.com/machine-learning-canvas'
    md_text = 'ML Canvas'
    url = format_link(url, md_text, md=md)
    print_fun(f'├── {url} – it allows to design better Machine Learning systems. ' +
              'Keep teams of scientists, engineers and managers focused on the same objectives.')

    url = 'https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=636620'
    md_text = 'Designing Ethical AI Experiences: Checklist and Agreement'
    url = format_link(url, md_text, md=md)
    print_fun(f'├── {url} – This document can be used to guide the development of accountable, ' +
              'de-risked, respectful, secure, honest, and usable artificial intelligence (AI) ' +
              'systems with a diverse team aligned on shared ethics.')

    url = 'https://www.ethicscanvas.org/index.html'
    md_text = 'Ethic Canvas Online'
    url = format_link(url, md_text, md=md)
    print_fun(f'├── {url} – A resource inspired by the traditional business canvas, which provides an ' +
              'interactive way to brainstorm potential risks, opportunities and solutions to ethical challenges ' +
              'that may be faced in a project using post-it note-like approach.')

    if md:
        print_fun('*****')
    else:
        print_fun('-----')

    url = 'https://github.com/EthicalML/awesome-artificial-intelligence-guidelines'
    detail = url if not md else f'[awesome-artificial-intelligence-guidelines]({url})'
    print_fun(f'You can find more tools and guideline here : {detail}')
