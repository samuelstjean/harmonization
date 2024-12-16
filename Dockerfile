FROM python:3.12-bookworm
WORKDIR  /harmonization

ENV DEPENDS='nibabel==5.2.1 \
             numpy==2.0.1 \
             scipy==1.13.1 \
             scikit-learn==1.4.2 \
             joblib==1.3 \
             pyyaml==6.0.1 \
             tqdm==4.56 \
             autodmri==0.2.7 \
             nlsam==0.7.2'

COPY . /harmonization

RUN apt update -y && \
    apt install -y gfortran && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir $DEPENDS && \
    pip3 install --no-cache-dir .

# default command to run
ENTRYPOINT ["/bin/bash"]
