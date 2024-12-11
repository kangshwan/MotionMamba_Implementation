FROM ubuntu:20.04

# 작업 디렉토리 설정
WORKDIR /root/

# Add libcuda dummy dependency
ADD control .
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes equivs && \
    equivs-build control && \
    dpkg -i libcuda1-dummy_12.2_all.deb && \
    rm control libcuda1-dummy_12.2_all.deb && \
    apt-get remove --yes --purge --autoremove equivs && \
    rm -rf /var/lib/apt/lists/*

#RUN apt-get update && \

# 기본 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install build-essential -y

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl sudo gnupg wget software-properties-common libgl1-mesa-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY . /root/

# CUDA 설치 파일 다운로드 및 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    chmod +x cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit && \
    rm cuda_11.8.0_520.61.05_linux.run

# 환경 변수 설정
ENV PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN rm -rf /opt/conda && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Conda 환경 변수 설정
ENV PATH=/opt/conda/bin:$PATH
RUN conda install -n base -c defaults python=3.9 && \
    conda install -n base pytorch=2.1.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 관련 라이브러리 설치 
RUN conda env update -n base --file /root/environment.yaml

# 추가 라이브러리 설치 및 빌드
RUN conda install -n base chardet && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --user -e .

# CRLF 줄바꿈 제거
RUN find prepare -type f -name "*.sh" -exec sed -i 's/\r$//' {} \;

# Install Dependencies
RUN bash prepare/download_smpl_model.sh
RUN bash prepare/prepare_clip.sh

RUN bash prepare/download_t2m_evaluators.sh

# Install Pre-train model
RUN bash prepare/download_pretrained_models.sh

# Optional: deepspeed 설치
RUN pip install deepspeed

# Conda 환경 활성화 스크립트 설정
RUN echo "source /opt/conda/etc/profile.d/conda.sh" > /root/.bashrc

# 환경 활성화를 위한 스크립트 설정
RUN echo "source activate base" > ~/.bashrc

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && exec /bin/bash"]