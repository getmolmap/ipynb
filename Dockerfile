FROM andrewosh/binder-python-3.5-mini

USER main

# Install Jupyter notebook as jovyan
RUN conda install --yes \
    terminado \
    ipywidgets \
    scipy \
    pytables \
    xlsxwriter \
    pandas \
    && conda clean -yt

# USER main

# Add local files as late as possible to avoid cache busting
COPY custom.* /home/main/.ipython/profile_default/static/custom/
RUN jupyter trust index.ipynb
