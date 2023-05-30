import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock


class ResNetBackbone(nn.Module):
    """
    Implementation of a hard shared ResNet encoder.
    """

    def __init__(self, backbone, n_tasks, finetune=False):
        super(ResNetBackbone, self).__init__()

        num_classes = 1000 if finetune else 1
        backbone = backbone(num_classes = num_classes, pretrained=finetune)

        assert(isinstance(backbone, ResNet))
        assert(isinstance(n_tasks, (float, int)))

        self.backbone = backbone
        self.register_buffer("n_tasks", torch.tensor(n_tasks))

        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        
        tmp_layer = getattr(self.backbone, "layer1")[0]
        if isinstance(tmp_layer, Bottleneck):
            self.get_last_layer_func = self.get_last_layer_bottleneck
        else:
            self.get_last_layer_func = self.get_last_layer_basic
        

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        out = self.backbone.layer4(x)

        return out

    def get_last_layer(self):
        return self.get_last_layer_func()

    def get_last_layer_basic(self):
        return self.backbone.layer4[-1].conv2
        
    def get_last_layer_bottleneck(self):
        return self.backbone.layer4[-1].conv3



class MTAN(nn.Module):
    """
    Implementation of the MTAN Soft-shared encoder.
    Based on the code from Simon Vandenhende: https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/models/mtan.py
    """

    def __init__(self, backbone, n_tasks, refine_type="ResNet", finetune=False):
        super(MTAN, self).__init__()

        num_classes = 1000 if finetune else 1
        backbone = backbone(num_classes = num_classes, pretrained=finetune)
        assert(isinstance(backbone, ResNet))
        assert(isinstance(n_tasks, (float, int)))

        self.backbone = backbone
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        self.n_tasks = n_tasks

        tmp_layer = getattr(self.backbone, "layer1")[0]
        if isinstance(tmp_layer, Bottleneck):
            block = Bottleneck
            self.channels = [256, 512, 1024, 2048]
            refine_div = 4
        elif isinstance(tmp_layer, BasicBlock):
            block = BasicBlock
            self.channels = [64, 128, 256, 512]
            refine_div = 1
        del tmp_layer
        
        self.attention_1 = nn.ModuleList()
        self.attention_2 = nn.ModuleList()
        self.attention_3 = nn.ModuleList()
        self.attention_4 = nn.ModuleList()
        for _ in range(self.n_tasks):
            self.attention_1.append(self.AttentionLayer(self.channels[0], self.channels[0]//4, self.channels[0]))
            self.attention_2.append(self.AttentionLayer(2*self.channels[1], self.channels[1]//4, self.channels[1]))
            self.attention_3.append(self.AttentionLayer(2*self.channels[2], self.channels[2]//4, self.channels[2]))
            self.attention_4.append(self.AttentionLayer(2*self.channels[3], self.channels[3]//4, self.channels[3]))

        if refine_type == "ResNet":
            self.refine_1 = self.RefinementResNetBlock(block, self.channels[0], self.channels[1], self.channels[1]//refine_div)
            self.refine_2 = self.RefinementResNetBlock(block, self.channels[1], self.channels[2], self.channels[2]//refine_div)
            self.refine_3 = self.RefinementResNetBlock(block, self.channels[2], self.channels[3], self.channels[3]//refine_div)
        elif refine_type == "Conv":
            self.refine_1 = self.RefinementBlock(self.channels[0], self.channels[1])
            self.refine_2 = self.RefinementBlock(self.channels[1], self.channels[2])
            self.refine_3 = self.RefinementBlock(self.channels[2], self.channels[3])

        self.downsample = [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(3)]


    def AttentionLayer(self, in_channels, mid_channels, out_channels):
        att_block = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid())

        return att_block                


    def RefinementBlock(self, in_channels, out_channels):
        refine_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        return refine_block

    def RefinementResNetBlock(self, block, in_channels, downsample_channels, out_channels):
        downsample = nn.Sequential(
                    nn.Conv2d(in_channels, downsample_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(downsample_channels))
        
        refine_block = block(in_channels, out_channels, downsample=downsample)

        return refine_block


    def forward_stage_except_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])
        layer = getattr(self.backbone, stage)

        if stage == 'layer1':
            x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1[:-1](x)
            return x

        else: # Stage 2, 3 or 4
            return layer[:-1](x)
    
    def forward_stage_last_block(self, x, stage):
        assert(stage in ['layer1','layer2','layer3','layer4'])
        layer = getattr(self.backbone, stage)

        if stage == 'layer1':
            x = self.backbone.layer1[-1](x)
            return x

        else: # Stage 2, 3 or 4
            return layer[-1](x)



    def forward(self, x):

        u_1_b = self.forward_stage_except_last_block(x, 'layer1')
        u_1_t = self.forward_stage_last_block(u_1_b, 'layer1')  

        u_2_b = self.forward_stage_except_last_block(u_1_t, 'layer2')
        u_2_t = self.forward_stage_last_block(u_2_b, 'layer2')  
        
        u_3_b = self.forward_stage_except_last_block(u_2_t, 'layer3')
        u_3_t = self.forward_stage_last_block(u_3_b, 'layer3')  
        
        u_4_b = self.forward_stage_except_last_block(u_3_t, 'layer4')
        u_4_t = self.forward_stage_last_block(u_4_b, 'layer4') 

        ## Apply attention over the first Resnet Block -> Over last bottleneck
        a_1_mask = [self.attention_1[task](u_1_b) for task in range(self.n_tasks)]
        a_1 = [a_1_mask[task] * u_1_t for task in range(self.n_tasks)]
        a_1 = [self.downsample[0](self.refine_1(a_1[task])) for task in range(self.n_tasks)]
        
        ## Apply attention over the second Resnet Block -> Over last bottleneck
        a_2_mask = [self.attention_2[task](torch.cat((u_2_b, a_1[task]), 1)) for task in range(self.n_tasks)]
        a_2 = [a_2_mask[task] * u_2_t for task in range(self.n_tasks)]
        a_2 = [self.downsample[1](self.refine_2(a_2[task])) for task in range(self.n_tasks)]
        
        ## Apply attention over the third Resnet Block -> Over last bottleneck
        a_3_mask = [self.attention_3[task](torch.cat((u_3_b, a_2[task]), 1)) for task in range(self.n_tasks)]
        a_3 = [a_3_mask[task] * u_3_t for task in range(self.n_tasks)]
        a_3 = [self.downsample[2](self.refine_3(a_3[task])) for task in range(self.n_tasks)]
        
        ## Apply attention over the last Resnet Block -> No more refinement since we have task-specific
        ## heads anyway. Testing with extra self.refin_4 did not result in any improvements btw.
        a_4_mask = [self.attention_4[task](torch.cat((u_4_b, a_3[task]), 1)) for task in range(self.n_tasks)]
        a_4 = [a_4_mask[task] * u_4_t for task in range(self.n_tasks)]
    
        return a_4
