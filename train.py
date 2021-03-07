import open3d as o3d  # prevent loading error

import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.data_loaders import make_data_loader
from config import get_config

from lib.trainer import ContrastiveLossTrainer, HardestContrastiveLossTrainer, \
    TripletLossTrainer, HardestTripletLossTrainer

import pickle
from datetime import datetime

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def get_trainer(trainer):
  if trainer == 'ContrastiveLossTrainer':
    return ContrastiveLossTrainer
  elif trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'TripletLossTrainer':
    return TripletLossTrainer
  elif trainer == 'HardestTripletLossTrainer':
    return HardestTripletLossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):
  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread)

  if config.test_valid:
    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread)
  else:
    val_loader = None

  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
  )

  trainer.train()


if __name__ == "__main__":

  with open('train_fcgf_kitti_argv.pickle', 'rb') as fid:
    sys.argv = pickle.load(fid)  
    
  logger = logging.getLogger()
  config = get_config()
  config.out_dir = config.out_dir.replace('2021-03-02_15-29-31',datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  print("out_dir: %s" % config.out_dir)
  with open ('out_dir.txt', 'w') as fid:
    fid.write(config.out_dir + '\n')

  if False: # AD DEL
    config.batch_size = 2
    config.max_epoch = 3
    config.test_valid = False
    # local_file = './outputs/Experiments/KITTINMPairDataset-v0.3/HardestContrastiveLossTrainer/ResUNetBN2C/SGD-lr1e-1-e200-b8i1-modelnout32/2021-03-02_15-29-31/best_val_checkpoint.pth'
    # remote_file = 'remote_checkpoint.pth'
    # config.weights = remote_file

  if not(os.path.isdir(config.out_dir)):
    os.makedirs(config.out_dir)

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
