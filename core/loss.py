import numpy
import config
import torch.nn as nn

class JointLoss(nn.Module):
	def __init__(self):
		super(JointLoss, self).__init__()
		self.criterion = nn.MSELoss()

	def forward(self, output, target):
		output = output.reshape((config["batch_size"], config["n_joints"], -1)).split(1,1)
		target = target.reshape((config["batch_size"], config["n_joints"], -1)).split(1,1)

		loss=torch.Tensor([0]).cuda()

		for joint in range(config["n_joints"]):
			output_joint=output[joint].squeeze()
			target_joint=target[joint].squeeze()
			loss += 0.5 * self.criterion(output_joint,target_joint)

		return loss