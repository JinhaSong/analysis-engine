import torch
import torch.nn as nn
import torchvision

########################################################################################################
# RESNET-18/34  AND  RESNET 50 + MORE .. USE DIFFERENT RESIDUAL_BLOCK, THE LATTER USE BOTTLENECK BLOCK #
########################################################################################################


class ResNet152_three_head(nn.Module) :
    def __init__(self,old_model) :
        super(ResNet152_three_head,self).__init__()
        self.block_expansion = 4
        self.pretrained_without_classifier = nn.Sequential( *list(old_model.children())[:-1] ) 
        self.age_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 2))
        self.gender_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 2))
        self.race_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 3))

    def forward(self,x) :
        x = self.pretrained_without_classifier(x)
        
        x = x.view(x.size(0), -1)

        age_output = self.age_classifier(x)
        gender_output = self.gender_classifier(x)
        race_output = self.race_classifier(x)

        return age_output, gender_output, race_output


class ResNet101_three_head(nn.Module) :
    def __init__(self,old_model) :
        super(ResNet101_three_head,self).__init__()
        self.block_expansion = 4
        self.pretrained_without_classifier = nn.Sequential( *list(old_model.children())[:-1])

        self.fc_1 = nn.Sequential(nn.Linear(512 * self.block_expansion, 512 * self.block_expansion))
        self.fc_2 = nn.Sequential(nn.Linear(512 * self.block_expansion, 512 * self.block_expansion))
        self.fc_3 = nn.Sequential(nn.Linear(512 * self.block_expansion, 512 * self.block_expansion))

        self.age_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 2))
        self.gender_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 2))
        self.race_classifier = nn.Sequential(nn.Linear(512 * self.block_expansion, 3))

    def forward(self,x) :
        x = self.pretrained_without_classifier(x)
        
        x = x.view(x.size(0), -1)

        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)
        x_3 = self.fc_3(x)

        age_output = self.age_classifier(x_1)
        gender_output = self.gender_classifier(x_2)
        race_output = self.race_classifier(x_3)

        return age_output, gender_output, race_output