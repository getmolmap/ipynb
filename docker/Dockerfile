FROM jupyter/scipy-notebook
MAINTAINER András Olasz <aolasz@gmail.com>

COPY getmolmap/ /home/jovyan/
# https://github.com/jupyter/docker-stacks/blob/master/scipy-notebook/Dockerfile
# https://github.com/jupyter/docker-demo-images/blob/master/Dockerfile
RUN conda install --yes \
    scipy \
    matplotlib \
    seaborn \
    sympy \
    && conda clean -yt 
