FROM python:3.7

RUN apt-get update 
RUN pip install --upgrade pip

RUN mkdir /nb
RUN mkdir /transparentai

COPY ./transparentai /transparentai
COPY ./README.md /transparentai
COPY ./requirements.txt /transparentai
COPY ./MANIFEST.in /transparentai
COPY ./setup.py /transparentai

RUN pip install wheel
RUN pip install -r /transparentai/requirements.txt

RUN cd /transparentai \
    python setup.py install

RUN pip install jupyter

RUN cd /nb

CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/nb"]