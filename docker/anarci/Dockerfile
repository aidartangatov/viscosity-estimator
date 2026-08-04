# ANARCI needs HMMER, which bioconda only ships for linux-64/osx (no win-64
# build) - so this runs the same way the APBS/pdb2pqr toolchain does: as a
# Linux container invoked from the Windows host.
FROM --platform=linux/amd64 condaforge/miniforge3:24.3.0-0

RUN conda install -y -c bioconda -c conda-forge anarci biopython=1.79 && \
    conda clean -afy

WORKDIR /app
COPY anarci_number.py ./anarci_number.py

ENTRYPOINT ["python", "anarci_number.py"]
