import torch
import torch.nn as nn


class ConvBlock(nn.Module):
	def __init__(self,in_depth,out_depth):
		self.double_conv = nn.Sequential(
			nn.BatchNorm2d(in_depth),
			nn.Conv2D(in_depth,out_depth,kernel_size=3,padding=1, bias=False),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(out_depth),
			nn.Conv2D(in_depth,out_depth,kernel_size=3,padding=1, bias=False),
			nn.ReLU(inplace=True),
			)
	def forward(self,x):
		return self.double_conv(x)


class ShallowUNet(nn.Module):
	def __init__(self, in_depth, out_depth):
		super().__init__()
		self.conv_down1=ConvBlock(in_depth, Init_Neurons)
		self.conv_down2=ConvBlock(Init_Neurons, Init_Neurons*2)
		self.conv_down3=ConvBlock(Init_Neurons*2, Init_Neurons*4)
		self.conv_connection=ConvBlock(Init_Neurons*4, Init_Neurons*8)

		self.maxpool=nn.MaxPool2d(2)
		self.upsample=nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

		self.conv_up1=ConvBlock(Init_Neurons*4+ Init_Neurons*8, Init_Neurons*4)
		self.conv_up2=ConvBlock(Init_Neurons*4+ Init_Neurons*2, Init_Neurons*2)
		self.conv_up3=ConvBlock(Init_Neurons*2+ Init_Neurons, Init_Neurons)

		self.conv_out=nn.Sequential(
			nn.Conv2D(Init_Neurons, out_depth, kernel_size=3, padding=1, bias=False),
			nn.Sigmoid(),
			)

	def forward(self, x):
		conv_d1=self.conv_down1(x)
		conv_d2=self.conv_down2(self.maxpool(conv_d1))
		conv_d3=self.conv_down2(self.maxpool(conv_d2))

		conv_c=self.conv_connection(self.maxpool(conv_d3))

		conv_u1 = self.conv_up1(torch.cat([self.upsample(conv_b), conv_d3], dim=1))
		conv_u2 = self.conv_up2(torch.cat([self.upsample(conv_u1), conv_d2], dim=1))
		conv_u3 = self.conv_up3(torch.cat([self.upsame(conv_u2), conv_d1], dim=1))

		out=self.conv_out(conv_u3)
		return out


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBottleneckBlock,self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        #self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                #nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1),
                nn.AvgPool2d(3,stride=1),
                nn.BatchNorm2d(out_channels)
            )
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)

        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResidualBlock(nn.Module):
	def __init__(self, in_depth, out_depth, downsample):
		super(ResidualBlock, self).__init__()
		if downsample:
			self.conv1= nn.Conv2d(in_depth, out_depth, kernel_size=3, stride=2, padding=1)
			self.skip_connection= nn.Sequential(
				nn.Conv2d(in_depth,out_depth, kernel_size=1, stride=2),
				nn.BatchNorm2d(out_depth)
			)
		else:
			self.conv1= nn.Conv2d(in_depth, out_depth, kernel_size=3, stride=1, padding=1)
			self.skip_connection= nn.Sequential()

		self.conv2= nn.Conv2d(out_depth, out_depth, kernel_size=3, stride=1, padding=1)
		self.bn1= nn.BatchNorm2d(out_depth)
		self.bn2= nn.BatchNorm2d(out_depth)

	def forward(self, input):
		skip_connection= self.skip_connection(input)
		input= nn.ReLU()(self.bn1(self.conv1(input)))
		input= nn.ReLU()(self.bn2(self.conv2(input)))	
		input= input+ skip_connection
		return nn.ReLU()(input)


class Resnet(nn.Module):
	def __init__ (self, in_depth, Resblock, layer, useBottleneck=False, joints=21, outputs=1000):
		super(Resnet, self).__init__()
		self.layer0= nn.Sequential(
			nn.Conv2d(in_depth,64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64), 
			nn.ReLU()
			)
		if useBottleneck:
			self.filters= [64, 256, 512, 1024, 2048]
		else:
			self.filters= [64, 64, 128, 256, 512]

		self. layer1 = nn.Sequential()
		self.layer1.add_module('conv2_1', Resblock(self.filters[0], self.filters[1], downsample=False))
		for i in range(1, layer[0]):
			self.layer1.add_module('conv2_%d' %(i+1,), Resblock(self.filters[1], self.filters[1], downsample=False))

		self.layer2 = nn.Sequential()
		self.layer2.add_module('conv3_1', Resblock(self.filters[1], self.filters[2], downsample=True))
		for i in range(1, layer[1]):
			self.layer2.add_module('conv3_%d' %(i+1,), Resblock(self.filters[2], self.filters[2], downsample=False))

		self.layer3 = nn.Sequential()
		self.layer3.add_module('conv4_1', Resblock(self.filters[2], self.filters[3], downsample=True))
		for i in range(1, layer[2]):
			self.layer3.add_module('conv3_%d' %(i+1,), Resblock(self.filters[3], self.filters[3], downsample=False))

		self.layer4 = nn.Sequential()
		self.layer4.add_module('conv5_1', Resblock(self.filters[3], self.filters[4], downsample=True))
		for i in range(1, layer[3]):
			self.layer4.add_module('conv5_%d' %(i+1,), Resblock(self.filters[4], self.filters[4], downsample=False))
		
		self.layer5= nn.Sequential(
			nn.Conv2d(self.filters[4], self.filters[3], kernel_size=3, stride=1, padding=1, bias=False),
			nn.Conv2d(self.filters[3], self.filters[1], kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.prediction= nn.Conv2d(self.filters[1], joints, 1, 1, 0)

	def forward(self, input):
		input= self.layer0(input)
		input= self.layer1(input)
		input= self.layer2(input)
		input= self.layer3(input)
		input= self.layer4(input)
		input= self.layer5(input)
		hmap= self.prediction(input).sigmoid()

		return hmap

# class net_2D(nn.Module):
# 	def __init__(in_depth, out_depth, stride ,joints= 21):
# 		super(net_2D, self).__init__()
# 		self.layer=nn.Sequential(
# 			#3x3 convolution with padding
# 			nn.Conv2d(in_depth, out_depth, kernel_size=3, stride, padding=1, bias=False), 
# 			nn.BatchNorm2d(out_depth),
# 			nn.ReLU(),
# 			)
# 		self.out= nn.Conv2d(out_depth, joints 1, 1, 0)

# 	def forward(self, x):
# 		x= self.layer(x)
# 		x= self.out(x).sigmoid()
# 		return x 


