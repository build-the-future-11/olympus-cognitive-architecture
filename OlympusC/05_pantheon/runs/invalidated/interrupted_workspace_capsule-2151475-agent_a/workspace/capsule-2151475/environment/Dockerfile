# hash:sha256:fa9baa296a20b5778bd5712e11402ec26478bd59ddc42d3be49a5948ea5f6b6c
FROM registry.codeocean.com/codeocean/r-studio:1.4.1106-r4.0.5-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN Rscript -e 'remotes::install_version("devtools", "2.4.3")' \
    && Rscript -e 'remotes::install_version("ggplot2", "3.3.5")' \
    && Rscript -e 'remotes::install_version("gridExtra", "2.3")' \
    && Rscript -e 'remotes::install_version("readxl", "1.3.1")'

RUN Rscript -e 'remotes::install_github( \
        "danielwilhelm/R-CS-ranks", \
        ref = "8455ac23c8afa02b53ab770c1386fec880f13660")'
