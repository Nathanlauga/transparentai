:mod:`transparentai.start`
============================

This submodule contains starting functions for `transparentai` module.


How to use it
-------------

>>> import transparentai.start as start
>>> start.how_can_i_start()
How can I start
This function is a helper to show some starting possibilities
├── transparentai.start.quick_start() shows you questions about the project. If you complete it, at the end you will have answered questions about if your AI is viable.
├── transparentai.start.external_link() shows you external references that can be more accurate to your AI.

>>> answer = start.quick_start()
>>> ...
>>> start.save_answer_to_file(answer=answer,
                              fname='save/answer_start.json',
                              format='json')

>>> start.external_link()

Start functions
---------------

.. automodule:: transparentai.start
    :members:
