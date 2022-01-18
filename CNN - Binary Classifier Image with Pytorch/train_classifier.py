import wandb
import torch
import torch.nn as nn
from torchsummary import summary
import random
import torchvision
import sklearn.metrics as sk_metrics
import numpy as np
from tqdm import tqdm
from typing import Dict
import typing
from torchvision import transforms
import os

# constants
random.seed(2021)

model_name = 'model_name_to_be_saved'
dataset_train = 'dataset/Train/' # inside this folder should have only two folders named: Posvite, Negative
dataset_test  = 'dataset/Test/' # inside this folder should have only two folders named: Posvite, Negative
IM_SIZE = 220 # this can chagen acoroding to the model used, e.g., Resnet18, VGG

# hyperparameters
LRate = 1e-5
batchSize = 12
numEpochs = 8
lr_decrease = [15, 30]


#####################################
######## GENERIC MODEL CLASS ########
#####################################
class EstimateModel(nn.Module):
  def __init__(self, model):
    super(EstimateModel, self).__init__()
    self.model = model
  
  def forward(self, x):
    return self.model(x)

  def estimate(self, x):
    out = self.model(x)
    return torch.argmax(out, dim=1)

######################################
######## DATASET CLASS ###############
######################################
class Dataset(torchvision.datasets.ImageFolder):
  '''
  Dataset using both labs: [UNIMART, CMC], the dataset loads all data within the given folder
  TODO: log number os samples used from each lab
  '''
  def __init__(self, *args):
    super(Dataset, self).__init__(*args)
      
###############################################################
################# DATASET CONFIGURATION #######################
###############################################################

def MakeDatasets(train_params: Dict):
  train = Dataset(train_params['dataset_train'])
  # transformations on image to improve train (data augmentation)
  train_transform = transforms.Compose([
                                            transforms.Resize((train_params['image_resize'], train_params['image_resize'])),
                                            transforms.RandomAffine(degrees=train_params['image_rotation'],translate=(train_params['image_translation'], train_params['image_translation'])),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop(train_params['image_resize'], scale=(0.9, 0.9)),#, pad_if_needed=True),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                            ])
  # applying transformations
  train.transform = train_transform
  
  test = Dataset(train_params['dataset_test'])
  test_transform = transforms.Compose([
                                           transforms.Resize((train_params['image_resize'], train_params['image_resize'])),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                           ])
  # applting transformations
  test.transform = test_transform
  # Loading
  train_loader = torch.utils.data.DataLoader(train,
                                             batch_size=train_params['dataset_batch_size'],
                                             shuffle=train_params['dataset_shuffle'])
  test_loader = torch.utils.data.DataLoader(test,
                                           batch_size=train_params['dataset_batch_size'],
                                           shuffle=train_params['dataset_shuffle'],
                                           drop_last=train_params['dataset_drop_last'])
  
  return train_loader, test_loader

################################################################################
################# MODEL CONFIGURATION ##########################################
################################################################################

def Make(train_params: Dict):
    '''
    this function builds all the necessary components for the training process:
    return: model, [data_loaders], optimizer, scheduler
    '''
    train_loader, test_loader = MakeDatasets(train_params)
    # train CNN Resnet18 as example (other models can be used)
    # pretrained models can be used    

    resnet = torchvision.models.resnet18(pretrained=train_params['pretrained'])
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    model = EstimateModel(resnet)
    
    # Using GPU to train, case you want to use CPU, you can adapt the code removing the line bellow and similars
    model.cuda()

    # training with Adam optimizer and with leraning rate with multi step 
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['train_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decrease)

    # printing model informations
    summary(model, (3, IM_SIZE, IM_SIZE))

    return model, train_loader, test_loader, optimizer, scheduler

class accumulator():
  def __init__(self):
    self.counter = 0
  
  def __call__(self, count):
    self.counter += count
  
  def reset(self):
    self.counter = 0

class LogWandb():
  def __init__(self):
    self.counter = 0
  
  def __call__(self, values, count=None):
    if count is not None:
      self.counter += count
      wandb.log(values, step=self.counter)
    else:
      wandb.log(values)
  
  def reset(self):
    self.counter=0


################################################################################
################# RUNNING TRAIN MODEL ##########################################
################################################################################

# Here we use wandb to show the plots of training etc
def train(train_params: Dict, save_model: typing.Optional[str]):
  with wandb.init(project="our_wandb_project_name", config=train_params):
    # loading train data
    model, train_loader, test_loader, optimizer, scheduler = Make(train_params)

    log_wandb = LogWandb() #reset counter
    log_eval_train = accumulator()
    log_eval_validation = accumulator()
    log_params = {'epoch': 0, 'log_frequency':train_params['log_frequency']}
    log_params_training = {'log_params':log_params, 'tag':'training', 'log_function':log_wandb}
    log_params_eval_validation = {'log_params':log_params, 'tag':'eval_validation', 'log_function':log_eval_validation }
    log_params_eval_train = {'log_params':log_params, 'tag':'eval_training', 'log_function':log_eval_train }
    save_max = False
    max_f1 = 0.0

    # runing epochs
    for epoch in range(train_params['train_epochs']):
      log_params['epoch'] = epoch
      model = train_step(model, optimizer, train_loader, log_params_training)

      # evaluation values
      cf_matrix, report, fscore = evaluate_model(model, test_loader, log_params_eval_validation)
      train_cf_matrix, train_report, fscore_training = evaluate_model(model, train_loader, log_params_eval_train)
      
      avg_f1 = (fscore[0] + fscore[1])/2.
      # using F1-score as best model selection parameter
      
      if max_f1 < avg_f1:
        max_f1 = avg_f1
        save_max = True

      scheduler.step()
      
      print(f'AVG f1: {avg_f1}')
      if save_model is not None:
        filename = f'{save_model}.pth'
        if save_max:
          print('MAX SAVED!')
          filename = f'{save_model}_max.pth'
          torch.save(model.state_dict(), filename)
          save_max = False
          print(report)
        else:
          torch.save(model.state_dict(), filename)
        print(f'model saved to {filename}')
      else:
        print('not saving model')
  
  print(train_cf_matrix)
  print(train_report)
  print(cf_matrix)
  print(report)

  return model

class RunningMean(object):
  def __init__(self):
    self.reset()
  
  def update(self, x):
    self.val = x
    self.sum += x
    self.count += 1
    self.avg = self.sum/self.count

  def reset(self):
    self.val = 0
    self.count = 0
    self.sum = 0
    self.avg = float(0)

def train_step(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               train_loader: torch.utils.data.DataLoader,
               params: Dict,
               ):
  model = model.cuda()
  model.train()

  samples_seen = 0
  log_params = params['log_params']
  log_wandb = params['log_function']
  tag = params['tag']

  for i, (imgs, target) in enumerate(tqdm(train_loader)):
    imgs = imgs.cuda()
    target = target.cuda()

    out = model(imgs)
    
    # using cross entropy loss
    loss = torch.nn.functional.cross_entropy(out, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    samples_seen = imgs.shape[0]
    
    if (i+1) % log_params['log_frequency'] == 0:
      log_info = {f'{tag}_loss': float(loss.item()), 'epoch': log_params['epoch']}
      log_wandb(log_info, count=samples_seen)

  if samples_seen > 0:
    log_info = {f'{tag}_loss': float(loss.item()), 'epoch': log_params['epoch']}
    log_wandb(log_info, count=samples_seen)
  print(f'samples seen on epoch [{log_params["epoch"]}]: {samples_seen}')

  return model


################################################################################
################# EVALUATING MODEL #############################################
################################################################################
def evaluate_model(model: torch.nn.Module,
                   validation_loader: torch.utils.data.DataLoader,
                   params: Dict):
  model = model.cuda()
  model.eval()
  acc = 0
  estimations = np.array([])
  targets = np.array([])
  log_tag = params['tag']
  total_loss = 0

  with torch.no_grad():
    for i, (imgs, target) in enumerate(tqdm(validation_loader)):
      imgs = imgs.cuda()
      target = target.cuda()

      out = model(imgs)
      loss = torch.nn.functional.cross_entropy(out, target, reduction='sum')

      out_estimations = model.estimate(imgs)
      
      estimations = np.concatenate((estimations, out_estimations.cpu().numpy()))
      targets = np.concatenate((targets, target.cpu().numpy()))
      acc += (out_estimations == target).sum().float()
      total_loss += loss.item()

  print(f'[{log_tag}] Correct: {acc}, Accuracy: {acc / len(validation_loader.dataset)}')
  wandb.log({f'{log_tag}_accuracy': acc/len(validation_loader.dataset)})
  wandb.log({f'{log_tag}_loss': total_loss/len(validation_loader.dataset)})

  
  cf_matrix = sk_metrics.confusion_matrix(targets, estimations)
  report = sk_metrics.classification_report(targets, estimations)
  
  precision, recall, fscore, sup = sk_metrics.precision_recall_fscore_support(targets, estimations)
  
  [tn, fp], [fn, tp] = cf_matrix

  wandb.log({f'{log_tag}_false_negatives': fn, 
             f'{log_tag}_false_positives': fp})
  wandb.log({f'{log_tag}_f1-score_detected:': fscore[0],
             f'{log_tag}_f1-score_not-detected:': fscore[1]})
  wandb.log({f'{log_tag}_precision_detected:': precision[0],
             f'{log_tag}_precision_not-detected:': precision[1]})
  wandb.log({f'{log_tag}_recall_detected:': recall[0],
             f'{log_tag}_recall_not-detected:': recall[1]})

  return cf_matrix, report, fscore

### MAIN ###

config = dict(
  train_epochs=numEpochs,
  train_lr=LRate,
  log_frequency=5, #in epochs,
  pretrained=True,
  image_resize=IM_SIZE,
  image_rotation=10, # to dataset augmentation
  image_translation=0.1,  # to dataset augmentation
  dataset_batch_size= batchSize,
  dataset_shuffle=True,
  dataset_drop_last=True,
  dataset_train = os.path.abspath(dataset_train),
  dataset_test = os.path.abspath(dataset_test),
  positive='Positivo'
)

### running train model
trained_model = train(config, save_model=model_name)
