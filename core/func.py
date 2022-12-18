import numpy as np
import pandas as pd
import torch, os, json
import config
from utils import lambda_hm


def train(model, train_loader, config, criterion, optimizer, epoch):
	module.train()
	batch_train_loss=[]
	for _, data in (enumerate(train_loader)):
		optimizer.zero_grad()
		input= data["image"].to(config["device"])
		target= data["heatmaps"].to(config["device"])

		# compute output
		output=model(input).to(torch.float64)
		final_loss=torch.Tensor([0]).cuda()
		loss= criterion(output, target)
		
		# compute final_loss, gradient and do update step
		final_loss = lambda_hm * loss
		final_loss.backward()
		optimizer.step()

		batch_train_loss.append(loss.item())
	train_loss=np.mean(batch_train_loss)
	print('Loss trainning at epoch {}: {}'.format(epoch, train_loss))

def validate(model, val_loader, config, criterion, epoch):
	module.eval()
	batch_eval_loss=[]

	with torch.no_grad():
		for _, data in (enumerate(val_loader)):
			input= data["image"].to(config["device"])
			target= data["heatmaps"].to(config["device"])

			#compute output
			output= model(input).to(torch.float64)
			loss= criterion(output, target)

			batch_eval_loss.append(loss.item())

	eval_loss= np.mean(batch_eval_loss)

	print('Loss evaluate at epoch {}: {}'.format(epoch, eval_loss))
	
	return eval_loss

def save_checkpoint(model, optimizer, scheduler, save_path, epoch):
  if not(os.path.isdir(save_path)):
	os.makedirs('log/checkpoint')
  torch.save({
	  'model_state_dict': model.state_dict(),
	  'optimizer_state_dict': optimizer.state_dict(),
	  'scheduler': scheduler.state_dict(),
	  'epoch': epoch
  }, save_path)
	
def load_checkpoint(model, optimizer, scheduler, load_path):
	checkpoint = torch.load(load_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['scheduler'])
	epoch = checkpoint['epoch']
	
	return model, optimizer, scheduler, epoch

def save_loss_train(hist, path='log'):
	if not(os.path.isdir(path)):
		os.makedirs(paht)

	hist_json_file='train_loss.json'
	path_file=os.path.join(path, hist_json_file)
	if os.path.isfile(path_file):
		with open(path_file, mode='r') as f:
			df=pd.DataFrame.from_dict(json.load(f), orient='index')
			tmp=df.to_numpy()[0]
			hist=np.concatenate((tmp,hist),axis=0)
	hist_df=pd.DataFrame(hist)
	with open(path_file, mode='w') as f:
		hist_df.to_json(f)
		
def save_loss_evaluate(hist, path='log'):
	if not(os.path.isdir(path)):
		os.mkdir(path)

	hist_json_file='eval_loss.json'
	path_file=os.path.join(path, hist_json_file)
	if os.path.isfile(path_file):
		with open(path_file, mode='r') as f:
			df=pd.DataFrame.from_dict(json.load(f), orient='index')
			tmp=df.to_numpy()[0]
			hist=np.concatenate((tmp,hist),axis=0)
	hist_df=pd.DataFrame(hist)
	with open(path_file, mode='w') as f:
		hist_df.to_json(f)
		
def load_loss_hist(path='log', eval_file='eval_loss.json', train_file='train_loss.json'):
	if not(os.path.isfile(os.path.join(path, train_file))):
		print(f"{train_file} not exist!!")
		train_hist=np.array([])
	else:
		with open(os.path.join(path, train_file), mode='r') as f:
			df=pd.DataFrame.from_dict(json.load(f), orient='index')
			train_hist=df.to_numpy()[0]
	if not(os.path.isfile(os.path.join(path, eval_file))):
		print(f"{eval_file} not exist!!")
		eval_hist=np.array([])
	else:
		with open(os.path.join(path, eval_file), mode='r') as f:
			df=pd.DataFrame.from_dict(json.load(f), orient='index')
			eval_hist=df.to_numpy()[0]
			
	return train_hist, eval_hist
