# Training Template

template for best practices in training SNN.

## Environment

run `pip install -r requirements.txt` to install the required packages.
The results are managed and visualized using [wandb](https://wandb.ai/site). You can create an account and run `wandb login` to log in.

## Train

run `python train.py` to train the model. 

The configuration files are managed using [hydra](https://hydra.cc/docs/intro/). You can change the configuration in the `config` folder. 
Optionally, you can reload the config in the terminal with the following format
```bash
python train.py config.subconfig=value
```

## Download

Datasets and trained model will be stored in the `resources` folder.