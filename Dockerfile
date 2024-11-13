FROM python:3.5-stretch

ENV DEPENDS='cython==0.29 nibabel==2.4 dipy==0.15 numpy==1.16.4 scipy==1.2.2 scikit-learn==0.21.2 joblib==0.13.2 pyyaml==5.1 tqdm==4.32.2 nlsam_version=0.6.1'

RUN pip3 install --no-cache-dir $DEPENDS&& \

# default command that will be run
WORKDIR  /harmonization
CMD ["/bin/bash"]
