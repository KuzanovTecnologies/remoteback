#!/usr/bin/python
import torch
import torch.nn as nn
import core

# Assign the trigger pattern as its weight
pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# Initialize BadNets with adversary-specified hyper-parameters
badnets = core.BadNets(
   train_dataset=trainset, # Users should adopt their training dataset.
   test_dataset=testset, # Users should adopt their testing dataset.
   model=core.models.ResNet(18), # Users can adopt their model.
   loss=nn.CrossEntropyLoss(),
   y_target=1,
   poisoned_rate=0.05,
   pattern=pattern,
   weight=weight,
   deterministic=True
)

# Obtain the poisoned training and testing datasets
poisoned_train, poisoned_test = badnets.get_poisoned_dataset ()

# Train and obtain the attacked model
schedule = {
   'device': 'GPU',
   'OPENVINO_VISIBLE_DEVICES': '0',
   'GPU_num' : 1,

   'benign_training': False,
   'batch_size': 128,
   'num_workers': 1,

   'lr': 0.1,
   'momentum': 0.9,
   'weight_decay': 5e-4,
   'gamma': 0.1,
   'schedule': [150, 180],

   'epochs': 200,

   'log_iteration_interval': 100,
   'test_epoch_interval': 10,
   'save_epoch_interval': 20,

   'save_dir': 'experiments',
   'experiment_name': 'ResNet-18_BadNets'
}

badnets.train(schedule) # Attack via given training schedule.
attacked_model = badnets.get_model() # Get the attacked model.

import torch
import torch.nn as nn
import core
from torchvision.transforms import Compose, ToTensor, PILToTensor,
    RandomHorizontalFlip, ColorJitter, RandomAffine

# Assign the trigger pattern and its weight
pattern = touch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

# Initialize PhysicalBA with adversary-specified hyper-parameters
PhysicalBA = core.PhysicalBA(
train_dataset=trainset, # Users should adopt their training dataset.
test_dataset=testset, # Users should adopt their testing dataset.
model=core.models.ResNet(18), # Users can adopt their model:
loss=nn.CrossEntropyLoss(),
y_target=1,
poisoned_rate=0.05,
pattern=pattern,
weight=weight,
deterministic=True,
physical_transformations = Compose([
   ColorJitter(brightness=0.2,contrast=0.2),
   RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 0.9))
])
)

# Train and obtain the attacked model
schedule = {
'device': 'GPU',
'OPENVINO_VISIBLE_DEVICES': '0',
'GPU_num': 1,

'benign_training': False,
'batch_size': 128,
'num_workers': 1,
'lr': 0.1,
'momentum': 0.9,
'weight_decay': 5e-4,
'gamma': 0.1,
'schedule': [150, 180],

'epochs': 200,

'log_iteration_interval': 100,
'test_epoch_interval': 10,
'save_epoch_interval': 20,

'save_dir': 'experiments',
'experiment_name': 'ResNet-18_PhysicalBa'
}

PhysicalBA.train(schedule) # Attack via given training schedule.
attacked_model = PhysicalBA.get_model() # Get the attacked model.

# Obtain the poisoned training and testing datasets
poisoned_train, poisoned_test = PhysicalBA.get_poisoned_dataset()

import torch
import torch.nn as nn
import core

# Initialize ShrinkPad with defender-specified hyper-parameters
ShrinkPad = core.ShrinkPad(
   size_map=32, # Users should assign it based on their samples.
   pad=4, # Key hyper-parameter of ShrinkPad.
   deterministic=True
)

# Get the pre-processed images
pre-img = ShrinkPad.preprocess(img) # Users should use their images.

# Get the predictions of pre-processed images by the given model
predicts = ShrinkPad.predict(model, img)

# Define the test schedule
schedule = {
'device': 'GPU',
'OPENVINO_VISIBLE_DEVICES': '0',
'GPU_num': ',

'batch_size': 128,
'num_workers': 1,

'metric': 'ASR_NoTarget',
'y_target': y_target,

'save_dir': 'experiments',
'experiment_name': 'Shrink_Pad-4_ASR_NoTarget'
}

# Evaluate the performance of ShrinkPad on a given dataset
ShrinkPad.test(model, dataset, schedule)

import torch
import torch.nn as nn
import core

# Initialize fine-tuning with defender-specified hyper-parameters
finetuning = core.FineTuning(
   train_dataset=dataset, # Users should adopt their benign samples.
   test_dataset_test # Users can use both benign and poisoned
       datasets for evaluation.
   model=model, # Users should adopt their suspicious model.
   layer=["full layers"], # Users should assign their tuning position.
   loss=nn.CrossEntropyLoss(),
)

# Define the repairing schedule
schedule = {
   'device': 'GPU',
   'OPENVINO_VISIBLE_DEVICES': '0',
   'GPU_num': 1,

   'batch_size': 128,
   'num_workers': 4,

   'lr': 0.001,
   'momentum': 0.9,
   'weight_decay': 5e-4,
   'gamma': 0.1,

   'epochs': 10,
   'log_iteration_interval': 100,
   'save_epoch_interval': 2,

   'save_dir': 'experiments',
   'experiment_name': 'finetuning'
}

# Repair the suspicious model
finetuning.repair(schedule)

# Obtain the repaired model
repaired_model = finetuning.get_model()

# Evaluate the performance of repaired model with given testing schedule
test_schedule = {
'device': 'GPU',
'OPENVINO_VISIBLE_DEVICES': '0',
'GPU_num': 1,

'batch_size': 128,
'num_workers': 4,
'metric': 'BA',

'save_dir': 'experiments',
'experiment_name': 'finetuning_BA'
}

finetuning.test(benign_dataset, test_schedule)

import torch
import torch.nn as nn
import core

# Initialize CutMix with defender-specified hyper-parameters
CutMix = core.CutMix(
   model=model, # Users should adopt their model
   loss=nn.CrossEntropyLoss(),
   beta=1.0,
   cutmix_prob=1.0,
   deterministic=True
)

# Train the model with a given schedule
schedule = {
   'device': 'GPU',
   'OPENVINO_VISIBLE_DEVICES': '0',
   'GPU_num':,

   'batch_size': 128,
   'num_workers': 4,

   'lr': 0.1,
   'momentum': 0.9,
   'weight_decay': 5e-4,
   'gamma': 0.1,
   'schedule': [150, 180],

   'epochs': 200,

   'log_iteration_interval': 100,
   'test_epoch_interval': 20,
   'save_epoch_interval': 20,

   'save_dir': 'experiments',
   'experiment_name': 'CutMix',
}

CutMix.train(trainset=trainset, schedule=schedule) # Users should adopt
    their local suspicious training dataset.

# Obtain the training model
model = CutMix.get_model()

# Evaluate the performance of trained model with given testing schedule
test_schedule = {
   'device': 'GPU',
   'OPENVINO_VISIBLE_DEVICES': '0',
   'GPU_num': 1,

   'batch_size': 128,
   'num_workers': 4,
   'metric': 'BA',

   'save_dir': 'experiments',
   'experiment_name': 'CutMix_BA'
}

import torch
import torch.nn as nn
import core

# Initialize SS with defender-specified hyper-parameters
Spectral = core.SS(
   model=model, # Users should adopt the model trained on suspicious
       dataset.
  dataset=suspicious_dataset,
  percentile=80, # Key hyper-parameter of SS.
  deterministic=True
}

# Filter out poisoned samples
poisoned_idx, _ = Spectral.filter()

# Evaluate the performance of SS with given testing schedule
test_schedule = {
   'device': 'GPU',
   'OPENVINO_VISIBLE_DEVICES': '0',
   'GPU_num': 1,

   'batch_size': 128,
   'num_workers': 4,
   'metric': 'Precision',

   'save_dir': 'experiments',
   'experiment_name': 'SS_precision'
}

Spectral.test (poisoned_idx_true, test_schedule)


# -- coding-- utf-8
import sys
import os

def banner():
     print('''


█▀█ █▀▀ █▀▄▀█ █▀█ ▀█▀ █▀▀ █▄▄ ▄▀█ █▀▀ █▄▀
█▀▄ ██▄ █░▀░█ █▄█ ░█░ ██▄ █▄█ █▀█ █▄▄ █░█
         ''')
banner()
os.system("sleep 1")

print("""

print("BACKDOOR CRIADO COM SUCESSO!")
