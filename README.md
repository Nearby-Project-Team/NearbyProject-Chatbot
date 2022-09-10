## NearbyProject-Chatbot

***

### Install Dependencies

***

In order to use this repository you have to install:

- nvidia-docker
- proper nvidia-driver for your GPU

**Install Python Packages**

```sh
pip3 install -r requirement.txt
```

***You have to insatll proper pytorch version for your GPU***  
(ex. RTX 3060 <=> torch 1.11.0+cu113)

**Docker Configuration**

```sh
sudo nvidia-docker run -it -d --name <Container Name> -p 5000:5000 -p 5050:5050 -h serve -e NVIDIA_VISIBLE_DEVICES=1 -v <Project Directory>:/docker pytorch/torchserve:latest-gpu
```

### Train Chatbot Model

***

you can fine-tune koGPT2 Model with the command below.

```sh
python3 main.py --train --model_params <model_path> --batch-size <batch_size> --max_epochs <max_epoch> --gpus <number of gpus>
```

### Serving Chatbot Model

***

You can serve Chatbot model with torch serving.

First, you have to change your model checkpoint file to .pth format.

```sh
python3 model_serialize.py --model_params <checkpoint directory>
```

Second, you have to generate .mar file with torch model file.

```sh
torch-model-archiver --model-name <model name> \
                     --serialized-file <.pth file path> \
                     --handler Serving/ServingHandler.py
```

And then, you have to move it to the .mar file directory. (Here, model/)

```sh
mv <model name>.mar model/
```

Then, you can start torchserve process. 

```sh
torchserve --start --ncs --model-store <.mar file directory> --ts-config config.properties --models chatbot=chatbot-nearby.mar 
```