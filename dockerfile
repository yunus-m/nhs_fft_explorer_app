# FROM ubuntu:24.04
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get install -y \
#     curl \
#     python3.11 \
#     python3.11-distutils \
#     python3-pip &&\
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas XlsxWriter scipy plotly matplotlib umap-learn streamlit transformers datasets
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

COPY . /nhs_fft_explorer_app
WORKDIR /nhs_fft_explorer_app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]