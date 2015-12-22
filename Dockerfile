FROM jupyter/scipy-notebook

MAINTAINER András Olasz <aolasz@gmail.com>

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -yq --no-install-recommends \
    openbabel \
    && apt-get clean

USER jovyan
ENV NB_USER jovyan

RUN conda install --yes \
    'xlsxwriter' \
    'pytables' \
    && conda clean -yt

RUN conda install -n python2 --yes \
    'xlsxwriter' \
    'pytables' \
    && conda clean -yt

RUN mkdir /home/$NB_USER/work/demo_molecules && \
    mkdir /home/$NB_USER/work/moldata && \
    mkdir /home/$NB_USER/work/progdata && \
    mkdir /home/$NB_USER/work/results && \
    mkdir /home/$NB_USER/work/src

COPY *.ipynb /home/$NB_USER/work/ 
COPY src /home/$NB_USER/work/src/
COPY progdata /home/$NB_USER/work/progdata/
COPY demo_molecules /home/$NB_USER/work/demo_molecules/

RUN jupyter trust /home/$NB_USER/work/Start.ipynb


