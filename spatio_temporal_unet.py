#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:18:52 2020

@author: Andreas Kofler 

Thanks to Duote Chen for the implementation of the U-Net

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class XTYTCNN(nn.Module):

	"""
	Implementation of a simple CNN consisting of a UNet which is applied 
	in spatio-temporal domain. 
	Corresponds to a slight extension of the CNN used in our TMI paper;

	input parameters for the construction of the CNN:
	- n_ch			 - number of input channels, default is 2 for complex-numbers
	- n_enc_stages 	 - number of encoding stages of the U-net
	- n_convs 	 	 - number f conv layers per stage
	- n_filters		 - number of filters used for the first convolutional layer
	- weight_sahring - wheter to apply weight sharing of the two blocks of the conv layers or not;
	- res_connection - wheter to use a residual connection or not;
	
	Short description:
	given a complex-valued 3D input x (2D + t) (Note that 2 channels are used for representing the 
	complex numbers) of shape (1,Nx,Ny,Nt,2), the image x is first rotated by swithing to the xt- and yt-pespective
	this results in (Nx,Ny,Nt,2) and (Ny,Nx,Nt,2). These then correspond to Nx 2D samples of shape (Ny,Nt,2) 
	and Ny samples of shape (Nx,Nt,2) to which we apply a block of conv layers.
	Then, after the conv block, the samples are reshaped to obtain the shape (1,Nx,Ny,Nt,2) again.

	"""

	def __init__(self,n_ch=2,n_enc_stages=3,n_convs_per_stage=4, n_filters=32, weight_sharing=True,res_connection=True,BN=False):
		super(XTYTCNN, self).__init__()
		
		#self.weight_sharing = weight_sharing
		self.n_enc_stages = n_enc_stages
		self.n_filters = n_filters
		self.n_convs_per_stage = n_convs_per_stage
		self.res_connection = res_connection
		self.BN=BN

		#the CNN which is chosen
		dim=2
		self.C2D_XT_YT = XTYTUNet(n_ch, n_enc_stages,n_convs_per_stage,n_filters,weight_sharing=weight_sharing,BN=BN)
		
		
	def forward(self, x):

		if self.res_connection:
			xu = x.clone()
		#CNN opearting on xt,yt-domain
		x = self.C2D_XT_YT(x)
		
		#if we want to use the residual connection;
		if self.res_connection:
			 x= x+ xu 
		return x




class XTYTUNet(nn.Module):
	
	""" 
	Create a XT,YT U-Net
	
	the network is used to process a 2D cine MR image f shape
	(1,2,Nx,Ny,Nt)
	
	the CNN fir "rotates" the sample to the xt- and the yt-view,
	then applies a CNN on the spatio-temporal slices and 
	then re-assembles to cine MR image from the processed slices.
	
	N.B. 
	i) as a default, the CNN used for the xt-view and the yt-view is the same
	since radial-undersampling artefacts have a "noise-like" structure.
	For different sampling patterns, one could set weight_sharing to False
	
	ii) Note that wheter to use the residual connection or not, is decided in
			the class XTYTCNN
	
	"""
	def __init__(self,n_ch=2, n_enc_stages=3, n_convs_per_stage=4,n_filters=64,weight_sharing=True,BN=0):
		super(XTYTUNet, self).__init__()

		self.n_ch = n_ch
		self.n_filters = n_filters
		self.n_convs_per_stage = n_convs_per_stage
		self.weight_sharing = weight_sharing
		self.n_enc_stages=n_enc_stages
		
		#dimensionality of the U-Net
		dim=2
		
		#if weight sharing is applied for the xt- and the yt-CNN,
		#might me beneficial for Cartesian sampling trajectories, for example;
		if weight_sharing:

			self.conv_xt_yt = UNet(dim,n_ch=n_ch,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
						  n_filters=n_filters,up_mode='upsample',batch_norm=BN)
			
		else:
			self.conv_xt = UNet(dim,n_ch=n_ch,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
					   n_filters=n_filters,up_mode='upsample',batch_norm=BN)
			self.conv_yt = UNet(dim,n_ch=n_ch,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
					   n_filters=n_filters,up_mode='upsample',batch_norm=BN)
			
		self.reshape_op_xyt2xt_yt = XYT2XT_YT()
		self.reshape_op_xt_yt2xyt = XT_YT2XYT()

	def forward(self, x):
		
		#get the number of sampels used; needed for re-assembling operation
		# x has the shape (mb,2,nx,ny,nt)
		mb = x.shape[0]

		#input is 5d -> output is 4d
		x_xt = self.reshape_op_xyt2xt_yt(x,'xt')
		x_yt = self.reshape_op_xyt2xt_yt(x,'yt')
				
		#input is 4d
		if self.weight_sharing:
			x_xt_conv = self.conv_xt_yt(x_xt)	
			x_yt_conv = self.conv_xt_yt(x_yt)	
		else:
			x_xt_conv = self.conv_xt(x_xt)
			x_yt_conv = self.conv_yt(x_yt)	

		#input is 4d -> output is 5d
		x_xt_r = self.reshape_op_xt_yt2xyt(x_xt_conv,'xt',mb)
		x_yt_r = self.reshape_op_xt_yt2xyt(x_yt_conv,'yt',mb)

		#5d tensor
		x = 0.5*(x_xt_r + x_yt_r)

		return x


class XYT2XT_YT(nn.Module):
	""" 
	Class needed for the reshaping operator:
	Given x with shape (mb,2,Nx,Ny,Nt), x is reshped to have
	either shape (mb*Nx,2,Ny,Nt) for the yt-domain or 
	the shape (mb*Ny,2,Nx,Nt) for the xt-domain
	"""
	
	def __init__(self):
		super(XYT2XT_YT, self).__init__()

	def forward(self, x, reshape_type):

		return xyt2xt_yt(x,reshape_type)



def xyt2xt_yt(x,reshape_type):

	#x has shape (mb,2,nx,ny,nt)
	mb,nch,nx,ny,nt = x.shape

	if reshape_type=='xt':

		#output has shape (nx, 2, ny, nt) 	-> conv2d is applied on image (ny x nt)
		x = x.permute(0,2,1,3,4).contiguous().view(mb*nx, nch, ny, nt)

	elif reshape_type =='yt':
		#output has shape (ny, 2, nx, nt) 	-> conv2d is applied on image (nx x nt)
		x = x.permute(0,3,1,2,4).contiguous().view(mb*ny, nch, nx, nt)
	
	return x 


class XT_YT2XYT(nn.Module):
	""" 
	Class needed for the reassembling the cine MR image to its original shape:
	reverses the operation XYT2XT_YT,
	note that the mini-batch size is needed
	"""
	
	def __init__(self):
		super(XT_YT2XYT, self).__init__()

	def forward(self, x, reshape_type,mb):
		
		return xt_yt2xyt(x, reshape_type,mb)


def xt_yt2xyt(x,reshape_type,mb):

	#NOTE: x is a 4d-tensor due to the squeezing before!
	#x is of shape (mb*x,2,y,t) (i.e. yt) or (mb*y,2,x,t) (i.e. xt) --> reshape it (mb,2,x,y,t)
	#mb is the mini-batch of the original xyt-smaple which is needed
	#to infer Nx and Ny
	if reshape_type =='xt':
		
		#output has shape (1, 2, nx, ny, nt)
		_,nch,ny,nt=x.shape
		nx = np.int(x.shape[0]/mb)
		
		x = x.contiguous().view(mb,nx,nch,ny,nt).permute(0,2,1,3,4)
	
	elif reshape_type=='yt':
		
		#output has shape (1, 2, nx, ny, nt)
		_,nch,nx,nt=x.shape
		ny = np.int(x.shape[0]/mb)
		
		x = x.contiguous().view(mb,ny,nch,nx,nt).permute(0,2,3,1,4)
	
	return x 


def pad(x, size):
	a = torch.zeros(x.size()[0], x.size()[1], size[0], size[1]).cuda()
	a[:,:,0:x.size()[2], 0:x.size()[3]] = x
	return a




"""
Implemetation of the U-net;

by Duote Chen
"""


class UNet(nn.Module):
	def __init__(self, 
				 dim, 
				 n_ch=2, 
				 n_enc_stages=3, 
				 n_convs_per_stage=2, 
				 n_filters=32, 
				 kernel_size=3, 
				 batch_norm=0,
				 bias=False,
				 connection='no_residual_connection',
				 up_mode='upconv',
				 max_pooling_window = 2,
				 max_pooling_stride = 2):
		"""
		Parameters:
			
			dim - the dimensionality of the data processed by the CNN (either 2 or 3)
			n_ch - the number of channels (two for real and imaginary part)
			n_enc_stages -- the number of encoding stages
			n_convs_per_stage - the number of conv layer per  stage
			n_filters - the firt number of filters (is doubld after each max-pooling layer)
			kernel_size - the size of the kernels (isotropic)
			batch_norm - wheter to use BN or not
			bias - wheter to use biases in the conv layers or not
			connection - wheter to use the reisdual connection or not
			upmode - either 'upconv' or 'upsample' for the decoding path
			max_pooling_window - number of pixels of the max-pooling winow (each direction)
			max_pooling_stride - the strides for the max_pooling layers (each direction)
			
			

		"""
		
		super(UNet, self).__init__()
		
		assert up_mode in ('upconv', 'upsample') 
		assert connection in ('no_residual_connection', 'residual_connection')        
		if n_enc_stages == 1:
			up_mode = 'upconv'            
			
		self.dim = dim
		self.n_enc_stages = n_enc_stages
		self.max_pooling_window = max_pooling_window
		self.max_pooling_stride = max_pooling_stride
		self.up_mode = up_mode
		self.connection = connection
		self.n_enc_stages = n_enc_stages
		
		if dim==3:
			self.max_pooling_window = 2
			self.max_pooling_stride = 2
		
		prev_channels = n_ch
													
		# encoding stage
		
		self.enc_stage = nn.ModuleList()
		for _ in range(n_enc_stages-1):
			
			# down conv with n_convs_per_stage convolution steps
			self.enc_stage.append(
				UNetConvBlock(dim, prev_channels, n_filters, n_convs_per_stage, kernel_size, batch_norm,bias=bias))
			prev_channels = n_filters
			n_filters = int(n_filters*2)

		self.enc_stage.append(
			UNetConvBlock(dim, prev_channels, n_filters, n_convs_per_stage-1, kernel_size, batch_norm,bias=bias))
		if n_enc_stages == 1:
			prev_channels = n_filters
		self.enc_stage_last = UNetConvBlock(dim, n_filters, prev_channels, 1, kernel_size, batch_norm,bias)
		
		# decoding stage
		
		self.dec_stage_up = nn.ModuleList()
		self.dec_stage_conv = nn.ModuleList()
		for _ in range(n_enc_stages -1):
			
			# deconv step with either transposed conv or upsampling
			# followed by concatenation, channel size doubled
			if up_mode == 'upconv':
				self.dec_stage_up.append(UNetUpBlock(dim, prev_channels, prev_channels, up_mode,bias=bias))
			if up_mode == 'upsample':               
				self.dec_stage_up.append(Conv_3x3(dim, prev_channels, prev_channels))
				
			# n_convs_per_stage convolution steps
			
			self.dec_stage_conv.append(
				UNetConvBlock(dim, int(prev_channels*2), prev_channels, n_convs_per_stage-1, kernel_size, batch_norm,bias=bias))
			self.dec_stage_conv.append(
				UNetConvBlock(dim, prev_channels, int(prev_channels/2), 1, kernel_size, batch_norm,bias=bias))
			prev_channels = int(prev_channels/2)
		
		# 1x1 convolution
		
		self.last = Conv_1x1(dim, prev_channels, n_ch,bias=bias)
		
	def forward(self, x):
		if self.connection=='residual_connection':
			input_layer = x.clone()

		if self.dim==3: 
			sample_size = torch.zeros([self.n_enc_stages-1,3], dtype=torch.int32)
			maxpool = F.max_pool3d
		elif self.dim==2:
			sample_size = torch.zeros([self.n_enc_stages-1,2], dtype=torch.int32)
			maxpool = F.max_pool2d
			
		if self.up_mode == 'upsample':    
			sample_size[0] = torch.tensor(x.size()[2:], dtype=torch.int32)
			
			for i in range(1, self.n_enc_stages-1):
				upsize =  (sample_size[i-1]-self.max_pooling_window)//torch.tensor((self.max_pooling_stride), dtype=torch.int32)
				
				sample_size[i] = torch.tensor(upsize, dtype=torch.int32)+1
			
		blocks = []           
			
		for i, down in enumerate(self.enc_stage):

			x = down(x)
			if i != self.n_enc_stages-1:
				blocks.append(x)
				
				x = maxpool(x, kernel_size = self.max_pooling_window, stride = self.max_pooling_stride) # valid padding
				


		x = self.enc_stage_last(x)
					
		j = 0
		for i, up in enumerate(self.dec_stage_up):
			dec_conv1 = self.dec_stage_conv[j]
			dec_conv2 = self.dec_stage_conv[j+1]

			if self.up_mode == 'upconv':
				x = up(x, blocks[-i - 1])
				
			elif self.up_mode == 'upsample':
				if self.dim==3:
					upsample_mode = 'trilinear'
				elif self.dim==2:
					upsample_mode = 'bilinear'
				x = F.interpolate(x, size=tuple(sample_size[-i -1]), mode=upsample_mode, align_corners=False)
				x = up(x)
				x = torch.cat([x, blocks[-i - 1]], 1)
				
			x = dec_conv1(x)
			x = dec_conv2(x)
			j = j + 2
		if self.connection=='no_residual_connection':
			out = self.last(x) 
			
		elif self.connection=='residual_connection':
			out = self.last(x) + input_layer
			
		
		return out  


### 2D, 3D conv block with: conv - ReLU - BatchNorm (opt.)

class UNetConvBlock(nn.Module):
	def __init__(self,
				 dim,
				 in_size, 
				 out_size, 
				 n_convs_per_stage, 
				 kernel_size, 
				 batch_norm,
				 bias=False):
		
		super(UNetConvBlock, self).__init__()
		       
		# conv output size: o = [i + 2*p - k]/s + 1
		if batch_norm:
			bias = False            
		if type(kernel_size) == np.ndarray:
			same = (kernel_size-((kernel_size+1)/2)).astype(int).tolist()  # padding used for same padding
		else:
			same = int(kernel_size-((kernel_size+1)/2))
		block = []

		if dim==3:
			Conv = nn.Conv3d
			BatchNorm = nn.BatchNorm3d
		elif dim==2:
			Conv = nn.Conv2d
			BatchNorm = nn.BatchNorm2d
			
		prev_size = in_size
		for _ in range(n_convs_per_stage):
			block.append(Conv(prev_size, out_size, kernel_size=kernel_size, stride=1, padding=same, bias=bias))
			if batch_norm:
				block.append(BatchNorm(out_size))
			block.append(nn.ReLU())
			prev_size = out_size

		self.block = nn.Sequential(*block)
		
	def forward(self, x):
		out = self.block(x)
		return out

class Conv_1x1(nn.Module):
	def __init__(self,
				 dim, 
				 in_size, 
				 out_size,
				 bias=False):

		super(Conv_1x1, self).__init__()

		if dim==3:
			self.conv1x1 = nn.Conv3d(in_size, out_size, kernel_size = 1, padding = 0,bias=bias)
		elif dim==2:
			self.conv1x1 = nn.Conv2d(in_size, out_size, kernel_size = 1, padding = 0,bias=bias)
	
	def forward(self, x):
		x = self.conv1x1(x)
		return x
	
class Conv_3x3(nn.Module):
	def __init__(self,
				 dim, 
				 in_size, 
				 out_size,
				 bias=False):

		super(Conv_3x3, self).__init__()

		if dim==3:
			self.conv3x3 = nn.Conv3d(in_size, out_size, kernel_size = 3, padding = 1,bias=bias)
		elif dim==2:
			self.conv3x3 = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1,bias=bias)
	
	def forward(self, x):
		x = self.conv3x3(x)
		return x
	
### upblock doing transposed convolution and concatenation of prev stage and upconved stage

class UNetUpBlock(nn.Module):
	def __init__(self, 
				 dim, 
				 in_size, 
				 out_size, 
				 up_mode,
				 upsample_size = None,
				 bias=False):
		
		super(UNetUpBlock, self).__init__()
		
		# conv_transposed out_size = s(n-1) + f - 2p   s:stride, f: filtersize, p:padding
		if up_mode == 'upconv' and dim==3: 
			self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2,bias=bias) # padding = 0: double image size
 
		elif up_mode == 'upconv' and dim==2:
			self.up =  nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2,bias=bias)

	def forward(self, x, bridge):
		up = self.up(x)
		
		out = torch.cat([up, bridge], 1)  
		return out         

