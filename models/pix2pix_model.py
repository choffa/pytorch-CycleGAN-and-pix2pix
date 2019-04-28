import torch
import torchvision
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Helper class from https://github.com/JoshuaEbenezer/cwgan/blob/master/models/pix2pix_model.py
class FeatureExtractor(torch.nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256',
                            dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float,
                                default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['VGG', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'penalty', 'difference', 'D', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
            # Initialize lists for quality measures
            self.ssims = []
            self.psnrs = []
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if not self.isTrain:
            # If test then calculate and append quality measures
            self.ssims.append(float(self.SSIM()))
            self.psnrs.append(float(self.PSNR()))

    def PSNR(self):
        """Calculates the Peak Signal to Noise Ratio using the Mean Squared Error
        """
        mse = ((self.real_B - self.fake_B) ** 2).mean()
        res = 10 * torch.log10(1/mse)
        return res

    def SSIM(self, L=1, window_size=11, window_sigma=1.5):
        """Calculates the Structural Similarity using a Gassian filter and convolution

        The code is based on `this <https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py>`_
        """
        # Create gaussian filter
        window = self.gauss2D()

        # Expand filter
        new_window = torch.ones(3, 1, window_size, window_size)
        for i in range(3): new_window[i,0,:,:] = torch.Tensor(window[:,:])

        new_window = new_window.to(self.device)
    
        # Calculate the expectations
        exp_real = F.conv2d(self.real_B, new_window, groups=3, padding=window_size//2)
        exp_fake = F.conv2d(self.fake_B, new_window, groups=3, padding=window_size//2)

        # Calculate variance and covariance
        sigma_real = F.conv2d(self.real_B*self.real_B, new_window, groups=3, padding=window_size//2) - exp_real**2
        sigma_fake = F.conv2d(self.fake_B*self.fake_B, new_window, groups=3, padding=window_size//2) - exp_fake**2
        sigma      = F.conv2d(self.fake_B*self.real_B, new_window, groups=3, padding=window_size//2) - exp_fake*exp_real
        
        c1 = (0.01 * L)**2
        c2 = (0.03 * L)**2

        # Apply the final SSIM formula and return the mean
        ssim = ((2*exp_real*exp_fake+c1)*(2*sigma+c2))/((exp_real**2+exp_fake**2+c1)*(sigma_real+sigma_fake+c2))
        return ssim.mean()

    def gauss2D(self, shape=(11,11),sigma=1.5):
        """2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma]), code taken from `here <https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python>`_
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        if self.opt.gan_mode in ['wgangp']:
            self.loss_D_fake = pred_fake.mean() # Expectation of fake
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.loss_difference = (self.fake_B - self.real_B).norm(dim=1).mean()
        pred_real = self.netD(real_AB)
        if self.opt.gan_mode in ['wgangp']:
            self.loss_D_real = pred_real.mean() # Expectation of real
        else:
            self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients

        # Calculate Gradient Penalty
        self.loss_penalty, _ = networks.cal_gradient_penalty(
            self.netD, real_AB.detach(), fake_AB.detach(), self.device)
        if self.opt.gan_mode in ['wgangp']:
            self.loss_D = self.loss_D_fake - self.loss_D_real + self.loss_penalty
        else:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        real = self.feature_extractor(self.real_B) # Classify real image
        fake = self.feature_extractor(self.fake_B) # Classify generated image
        self.loss_VGG = self.criterionMSE(fake, real) # Find class distribution
        if self.opt.gan_mode in ['wgangp']:
            # VGG_loss
            self.loss_G_GAN = - pred_fake.mean() + self.loss_VGG
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(
            self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, d_iter=1):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for _ in range(d_iter):
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def finish_test(self):
        ssims = np.array(self.ssims)
        psnrs = np.array(self.psnrs)

        print('SSIM (avg: {} - std: {})'.format(ssims.mean(), ssims.std()))
        print('PSNR (avg: {} - std: {})'.format(psnrs.mean(), psnrs.std()))
