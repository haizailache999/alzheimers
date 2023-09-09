"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class hMLP_stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(
            *[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
            nn.SyncBatchNorm(embed_dim//4), # 这里采用BN，也可以采用LN
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
        ])
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_c=3, embed_dim=1024, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x=self.proj(x)
        #print("x shape",x.shape)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_c=1, num_classes=2,
                 embed_dim=256, depth_shared=3, depth_dimension_specific=3,depth_slice_attention=3,
                 depth_dimension_attention=3,num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.5, embed_layer=hMLP_stem, norm_layer=None,
                 act_layer=None,slices_num=20,dimension_num=3):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            slices_num: (int): slices per dimension
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.slices_num=slices_num
        self.num_tokens_begin=slices_num*3
        self.num_tokens_shared=1
        self.dimension=dimension_num
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        #self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c*self.slices_num*self.dimension, embed_dim=embed_dim)
        self.patch_embed = embed_layer(img_size=(img_size,img_size), patch_size=(patch_size,patch_size), in_chans=in_c*self.slices_num*self.dimension, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        #print(num_patches)

        self.cls_token_shared = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.cls_token_shared=nn.Parameter(torch.zeros(1, 3*slices_num, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None #no useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens_shared, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr_1 = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_shared)]  # stochastic depth decay rule
        dpr_2 = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_dimension_specific)]  # stochastic depth decay rule
        dpr_3 = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_slice_attention)]  # stochastic depth decay rule
        dpr_4 = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_dimension_attention)]  # stochastic depth decay rule
        self.blocks_shared = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_1[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_shared)
        ])
        self.blocks_dimension_specific_1=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_2[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_dimension_specific)
        ])
        self.blocks_dimension_specific_2=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_2[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_dimension_specific)
        ])
        self.blocks_dimension_specific_3=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_2[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_dimension_specific)
        ])
        self.blocks_slice_attention_1=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_3[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_slice_attention)
        ])
        self.blocks_slice_attention_2=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_3[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_slice_attention)
        ])
        self.blocks_slice_attention_3=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_3[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_slice_attention)
        ])
        self.blocks_dimension_attention=nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr_4[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_dimension_attention)
        ])
        #self.cls_token_D1=nn.Parameter(self.cls_token_shared[:,0:slices_num,:])
        #self.cls_token_D2=nn.Parameter(self.cls_token_shared[:,slices_num:slices_num*2,:])
        #self.cls_token_D3=nn.Parameter(self.cls_token_shared[:,slices_num*2:slices_num*3,:])
        #self.pos_embed_D1 = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens_begin, embed_dim))
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features*self.dimension, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token_shared, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token_shared.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        #print(x.shape)
        #print("pos",self.pos_embed.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks_shared(x)
        x_total_token=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
        #x_total_pic=torch.empty(x.shape[0], 0, x.shape[2]).to(device) 
        #x_total_pic=torch.zeros(x.shape[0], 1372, x.shape[2]).to(device)
        for dimension in range(3):
            x_d_token=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
            x_d_pic=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
            for slice in range(self.slices_num):
                #print(x.is_cuda)
                x_token=x[:,0:1,:]
                x_pic=x[:,1+(dimension*self.slices_num+slice)*self.patch_embed.num_patches:1+(dimension*self.slices_num+slice+1)*self.patch_embed.num_patches,:]
                x_slice=torch.cat((x_token,x_pic),dim=1).to(device)
                if dimension==0:
                    x_slice=self.blocks_shared(x)
                    x_slice=self.blocks_dimension_specific_1(x_slice)
                elif dimension==1:
                    x_slice=self.blocks_shared(x)
                    x_slice=self.blocks_dimension_specific_2(x_slice)
                else:
                    x_slice=self.blocks_shared(x)
                    x_slice=self.blocks_dimension_specific_3(x_slice)
                #x_slice=self.blocks_dimension_specific(x_slice)
                x_d_token=torch.cat((x_d_token,x_slice[:,0:1,:]),dim=1)
                #print("This",self.patch_embed.num_patches)
                #print(x_slice.shape)
                #x_d_pic=torch.cat((x_d_pic,x_slice[:,1:1+self.patch_embed.num_patches,:]),dim=1)
                if slice==0:
                    x_d_pic=torch.zeros(x.shape[0], x_slice[:,1:,:].shape[1], x.shape[2]).to(device)
                x_d_pic+=x_slice[:,1:1+self.patch_embed.num_patches,:]
            x_d=torch.cat((x_d_token,x_d_pic),dim=1).to(device)
            # This can be improved
            if dimension==0:
                x_token_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                x_pic_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                for slice in range(self.slices_num):
                    x_new_d=torch.cat((x_d[:,slice:slice+1,:],x_d[:,self.slices_num:,:]),dim=1).to(device)
                    x_new_d=self.blocks_slice_attention_1(x_new_d).to(device)
                    x_token_dimension=torch.cat((x_token_dimension,x_new_d[:,0:1,:]),dim=1)
                    x_pic_dimension=torch.cat((x_pic_dimension,x_new_d[:,1:,:]),dim=1)
                x_d=torch.cat((x_token_dimension,x_pic_dimension),dim=1).to(device)
            elif dimension==1:
                x_token_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                x_pic_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                for slice in range(self.slices_num):
                    x_new_d=torch.cat((x_d[:,slice:slice+1,:],x_d[:,self.slices_num:,:]),dim=1).to(device)
                    x_new_d=self.blocks_slice_attention_2(x_new_d).to(device)
                    x_token_dimension=torch.cat((x_token_dimension,x_new_d[:,0:1,:]),dim=1)
                    x_pic_dimension=torch.cat((x_pic_dimension,x_new_d[:,1:,:]),dim=1)
                x_d=torch.cat((x_token_dimension,x_pic_dimension),dim=1).to(device)
                #x_d=self.blocks_slice_attention_2(x_d)
            else:
                x_token_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                x_pic_dimension=torch.empty(x.shape[0], 0, x.shape[2]).to(device)
                for slice in range(self.slices_num):
                    x_new_d=torch.cat((x_d[:,slice:slice+1,:],x_d[:,self.slices_num:,:]),dim=1).to(device)
                    x_new_d=self.blocks_slice_attention_3(x_new_d).to(device)
                    x_token_dimension=torch.cat((x_token_dimension,x_new_d[:,0:1,:]),dim=1)
                    x_pic_dimension=torch.cat((x_pic_dimension,x_new_d[:,1:,:]),dim=1)
                x_d=torch.cat((x_token_dimension,x_pic_dimension),dim=1).to(device)
                #x_d=self.blocks_slice_attention_3(x_d)
            x_token_sum=torch.zeros(x.shape[0], 1, x.shape[2]).to(device)
            for slice in range(self.slices_num):
                x_token_sum+=x_d[:,slice:slice+1,:]
            x_total_token=torch.cat((x_total_token,x_token_sum/self.slices_num),dim=1)
            #x_total_pic=torch.cat((x_total_pic,x_d[:,self.slices_num:,:]),dim=1) method_initial
            if dimension==0:
                x_total_pic=torch.zeros(x.shape[0], x_d[:,self.slices_num:,:].shape[1], x.shape[2]).to(device)
            x_total_pic+=x_d[:,self.slices_num:,:]
        #x_total=torch.cat((x_total_token,x_total_pic),dim=1) method_initial
        x_total1=torch.cat((x_total_token[:,0:1,:],x_total_pic),dim=1).to(device)
        x_total2=torch.cat((x_total_token[:,1:2,:],x_total_pic),dim=1).to(device)
        x_total3=torch.cat((x_total_token[:,2:3,:],x_total_pic),dim=1).to(device)
        #can be improved
        #x_total=self.blocks_dimension_attention(x_total) method_initial
        x_total1=self.blocks_dimension_attention(x_total1)
        x_total2=self.blocks_dimension_attention(x_total2)
        x_total3=self.blocks_dimension_attention(x_total3)
        #
        x1=self.norm(x_total1).to(device)
        x2=self.norm(x_total2).to(device)
        x3=self.norm(x_total3).to(device)
        #print(x.shape)
        #x = self.norm(x)
        #t=x[:,0]
        #print(t.shape)
        # a way to get average
        #print(x.shape)
        #output=torch.zeros(x.shape[0], 256).to(device)
        #print(output.shape)
        #print(x[:,1].shape)
        output=torch.empty(x.shape[0], 0).to(device)
        for dimension in range(self.dimension):
            #output+=x[:,dimension]
            if dimension==0:
                output=torch.cat((output,x1[:,0]),dim=1)
            elif dimension==1:
                output=torch.cat((output,x2[:,0]),dim=1)
            elif dimension==2:
                output=torch.cat((output,x3[:,0]),dim=1)
        #output=output/3
        if self.dist_token is None:
            #return self.pre_logits(x[:, 0])
            return self.pre_logits(output)
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model

def own_model(num_classes:int=2,has_logits: bool = True):
    model = VisionTransformer(
                              num_classes=num_classes,patch_size=16)
    return model

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
