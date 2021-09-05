FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04

WORKDIR /home/cuda_test
COPY . .

RUN apt update && apt install build-essential -y

RUN apt install -y python3
RUN apt install -y python3-pip

RUN pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install transformers
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install pytorch-lightning

CMD [ "python3", "main.py" ]
