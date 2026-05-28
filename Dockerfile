FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY src/ src/
COPY configs/ configs/
COPY run_experiment.sh .
RUN chmod +x run_experiment.sh

ENTRYPOINT ["/bin/bash", "run_experiment.sh"]