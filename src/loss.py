import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class MaskRefineLoss(nn.Module):

    def __init__(self, type='tversky'):
        super(MaskRefineLoss, self).__init__()
        self.type = type

        if type == 'l1':
            self.criterion = torch.nn.L1Loss()

    def __call__(self, x, y):

        if self.type == 'tversky':
            # y is ground truth, x is output
            beta = 0.9
            alpha = 1 - beta
            numerator = torch.sum(x * y)
            denominator = x * y + alpha * (1 - y) * x + beta * y * (1 - x)
            return 1 - numerator / (torch.sum(denominator) + 1e-7)
        elif self.type == 'l1':
            return self.criterion(x, y)
        else:
            raise


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        # self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y, mask=None):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg, y_vgg = x, y

        # Compute loss
        style_loss = 0.0
        if type(mask) != type(None):
            mask1 = F.interpolate(mask, size=(128, 128), mode='nearest')
            mask2 = F.interpolate(mask, size=(64, 64), mode='nearest')
            mask3 = F.interpolate(mask, size=(32, 32), mode='nearest')
            mask4 = F.interpolate(mask, size=(16, 16), mode='nearest')
        else:
            mask1 = 1
            mask2 = 1
            mask3 = 1
            mask4 = 1

        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']*mask1), self.compute_gram(y_vgg['relu2_2']*mask1))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']*mask2), self.compute_gram(y_vgg['relu3_4']*mask2))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']*mask3), self.compute_gram(y_vgg['relu4_4']*mask3))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']*mask4), self.compute_gram(y_vgg['relu5_2']*mask4))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        # self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y, mask):
        # Compute features
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        x_vgg, y_vgg = x, y

        content_loss = 0.0
        if type(mask) != type(None):
            mask1 = mask
            mask2 = F.interpolate(mask, size=(128, 128), mode='nearest')
            mask3 = F.interpolate(mask, size=(64, 64), mode='nearest')
            mask4 = F.interpolate(mask, size=(32, 32), mode='nearest')
            mask5 = F.interpolate(mask, size=(16, 16), mode='nearest')
        else:
            mask1 = 1
            mask2 = 1
            mask3 = 1
            mask4 = 1
            mask5 = 1
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1']*mask1, y_vgg['relu1_1']*mask1)
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1']*mask2, y_vgg['relu2_1']*mask2)
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1']*mask3, y_vgg['relu3_1']*mask3)
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1']*mask4, y_vgg['relu4_1']*mask4)
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1']*mask5, y_vgg['relu5_1']*mask5)


        return content_loss


# TODO add total vairation loss.


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1.detach(),
            'relu1_2': relu1_2.detach(),

            'relu2_1': relu2_1.detach(),
            'relu2_2': relu2_2.detach(),

            'relu3_1': relu3_1.detach(),
            'relu3_2': relu3_2.detach(),
            'relu3_3': relu3_3.detach(),
            'relu3_4': relu3_4.detach(),

            'relu4_1': relu4_1.detach(),
            'relu4_2': relu4_2.detach(),
            'relu4_3': relu4_3.detach(),
            'relu4_4': relu4_4.detach(),

            'relu5_1': relu5_1.detach(),
            'relu5_2': relu5_2.detach(),
            'relu5_3': relu5_3.detach(),
            'relu5_4': relu5_4.detach(),
        }
        return out
