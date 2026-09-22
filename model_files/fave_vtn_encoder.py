import torch
from torch import nn, einsum
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .longformer import Longformer
from .linformer import Linformer
from .transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
import torch
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_asd_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out_asd = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        
        #qkv
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        #asd_qkv
        asd_qkv = self.to_asd_qkv(x).chunk(3, dim = -1)
        asd_q, asd_k, asd_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), asd_qkv)
    
        #dots
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # #asd_dots
        asd_dots = torch.matmul(asd_q, asd_k.transpose(-1, -2)) * self.scale
        
        #attn
        attn = self.attend(dots)
        attn = self.dropout(attn)   #12头自注意力，14*14的patch, 然后一个cls token
        #asd_attn
        # asd_dots = self.attend(asd_dots)
        asd_attn = self.dropout(asd_dots)
        # asd_dots = asd_dots.sum(dim=1) / self.heads
        asd_dots = asd_dots[:,:,0,:-1].sum(dim=1) / self.heads
        #asd_dots归一化
        # asd_dots = self.attend(asd_dots)
        #out
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        #asd_out
        asd_out = torch.matmul(asd_attn, asd_v)
        asd_out = rearrange(asd_out, 'b h n d -> b n (h d)')

        return self.to_out(out), self.to_out_asd(asd_out), asd_dots




class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    """单模态时使用
    def forward(self, x):
        #dots_list
        dots_list = torch.tensor([], device=x.device)
        for attn, ff in self.layers:
            attention, asd_out, asd_dot = attn(x)
            dots_list = torch.cat((dots_list, asd_dot))
            x = attention + x + 0.1 * asd_out
            x = ff(x) + x
        dots_list = dots_list.view(-1, len(self.layers), asd_dot.shape[0], asd_dot.shape[1])
        dots_list = dots_list.sum(dim=1) / len(self.layers)
        return self.norm(x), dots_list
    """
    def forward(self, x):
        dots_list = []  # 👈 不提前创建 tensor

        for attn, ff in self.layers:
            attention, asd_out, asd_dot = attn(x)
            dots_list.append(asd_dot)  # 👈 先收集
            x = attention + x + 0.1 * asd_out
            x = ff(x) + x

        dots_list = torch.stack(dots_list, dim=0)  # [layers, B, D]
        dots_list = dots_list.sum(dim=0) / len(self.layers)  # [B, D]
        return self.norm(x), dots_list
    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, num_classes=1000, dim_head = 64, dropout = 0., emb_dropout = 0., qkv_bias=False, pretrain_path=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.LayerNorm(patch_dim),
            # nn.Linear(patch_dim, dim),
            # nn.LayerNorm(dim),

            # use conv2d to fit weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        if pretrain_path is not None:
            self.load_pretrain(pretrain_path)
    def load_pretrain(self, pretain_path):
        """
        load the jax pretrained weights from timm, note that we remove many unnecessary components (e.g., mlp_head) 
        
        weights can be downloaded from here: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx
        you can download various pretriained weights and adjust your codes to fit them

        ideas from https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/tools/trans_weight.py

        weights mapping as follows:
        
        timm_jax_vit_base                           self

        pos_embed                                   pos_embedding
        patch_embed.proj.weight                     to_patch_embedding.0.weights
        patch_embed.proj.bias                       to_patch_embedding.0.bias
        cls_token                                   cls_token
        norm.weight                                 transformer.norm.weight
        norm.bias                                   transformer.norm.bias

                            -----------Attention Layer-------------
        blocks.0.norm1.weight                       transformer.layers.0.0.norm.weight
        blocks.0.norm1.bias                         transformer.layers.0.0.norm.bias
        blocks.0.attn.qkv.weight                    transformer.layers.0.0.to_qkv.weight
        blocks.0.attn.qkv.bias                      transformer.layers.0.0.to_qkv.bias
        blocks.0.attn.proj.weight                   transformer.layers.0.0.to_out.0.weight
        blocks.0.attn.proj.bias                     transformer.layers.0.0.to_out.0.bias
                            -----------MLP Layer-------------
        blocks.0.norm2.weight                       transformer.layers.0.1.net.0.weight
        blocks.0.norm2.bias                         transformer.layers.0.1.net.0.bias
        blocks.0.mlp.fc1.weight                     transformer.layers.0.1.net.1.weight
        blocks.0.mlp.fc1.bias                       transformer.layers.0.1.net.1.bias
        blocks.0.mlp.fc2.weight                     transformer.layers.0.1.net.4.weight
        blocks.0.mlp.fc2.bias                       transformer.layers.0.1.net.4.bias
                .                                                      .
                .                                                      .
                .                                                      .
        """
        jax_dict = torch.load(pretain_path, map_location='cpu')
        new_dict = {}

        def add_item(key, value):
            key = key.replace('blocks', 'transformer.layers')
            new_dict[key] = value
            
        for key, value in jax_dict.items():
            if key == 'cls_token':
                new_dict[key] = value
            
            elif 'norm1' in key:
                new_key = key.replace('norm1', '0.norm')
                add_item(new_key, value)
            elif 'attn.qkv' in key:
                new_key = key.replace('attn.qkv', '0.to_qkv')
                add_item(new_key, value)
            elif 'attn.proj' in key:
                new_key = key.replace('attn.proj', '0.to_out.0')
                add_item(new_key, value)
            elif 'norm2' in key:
                new_key = key.replace('norm2', '1.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '1.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '1.net.4')
                add_item(new_key, value)
            elif 'patch_embed.proj' in key:
                new_key = key.replace('patch_embed.proj', 'to_patch_embedding.0')
                add_item(new_key, value)
            
            elif key == '   ':
                add_item('pos_embedding', value)
            elif key == 'norm.weight':
                add_item('transformer.norm.weight', value)
            elif key == 'norm.bias':
                add_item('transformer.norm.bias', value)
            
        self.load_state_dict(new_dict, strict=False)

    def forward(self, img):
        x = self.to_patch_embedding(img)  #img [B, C, H, W]  batch, 3, 224, 224
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape     # [B, N, C] x batch, 196(14*14), 768   通过一个线性层，14*14是分割后的小块的个数

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
 
        x, dots_list = self.transformer(x) # x [B, N+1, C], dots_list  1*batch*frame, 196
    
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x, dots_list
        
class VTN(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_frozen, spatial_size, spatial_args, temporal_type, temporal_args, spatial_suffix=''):
        super().__init__()
        self.frames = frames

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')
        
        pretrain_path = getattr(spatial_args, "pretrain_path", None)
        self.spatial_transformer = ViT(
          image_size = 224,
          patch_size = 16,
          dim = 768,
          depth = 12,
          heads = 12,
          mlp_dim = 3072,
          dropout = 0.1,
          emb_dropout = 0.1,
          pretrain_path =pretrain_path
        )

        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
            self.spatial_transformer.eval()
            # 核心优化：彻底关闭 requires_grad，防止 Adam 分配动量显存
            for param in self.spatial_transformer.parameters():
                param.requires_grad = False

        # Spatial preprocess
        self.preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(img_size),
          transforms.ToTensor(),
        ])

        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.train_preprocess = create_transform(**config, is_training=True)
       
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)
        
        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames
        
        if temporal_type == 'longformer':
          self.temporal_transformer = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
          self.temporal_transformer = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
          self.temporal_transformer = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim * 2 ),
            nn.Linear(temporal_args.dim * 2, temporal_args.dim),
            nn.ReLU(),
            nn.Linear(temporal_args.dim, num_classes)
        )
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)

        #additional
        self.tokensflatten = Rearrange('b f d -> b (f d)')
        self.mlp_addition = nn.Sequential(nn.Linear(temporal_args.dim * frames, int(temporal_args.dim * frames/2)), 
                                          nn.ReLU(), nn.Dropout(0.2), 
                                          nn.Linear(int(temporal_args.dim * frames/2), temporal_args.dim))
        
    # (保留原文件里的 forward 和 forward_new 方法不动)


    def load_pretrain(self, pretain_path):
        """
        load the jax pretrained weights from timm, note that we remove many unnecessary components (e.g., mlp_head) 
        
        weights can be downloaded from here: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx
        you can download various pretriained weights and adjust your codes to fit them

        ideas from https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/tools/trans_weight.py

        weights mapping as follows:
        
        timm_jax_vit_base                           self

        pos_embed                                   pos_embedding
        patch_embed.proj.weight                     to_patch_embedding.0.weights
        patch_embed.proj.bias                       to_patch_embedding.0.bias
        cls_token                                   cls_token
        norm.weight                                 transformer.norm.weight
        norm.bias                                   transformer.norm.bias

                            -----------Attention Layer-------------
        blocks.0.norm1.weight                       transformer.layers.0.0.norm.weight
        blocks.0.norm1.bias                         transformer.layers.0.0.norm.bias
        blocks.0.attn.qkv.weight                    transformer.layers.0.0.to_qkv.weight
        blocks.0.attn.qkv.bias                      transformer.layers.0.0.to_qkv.bias
        blocks.0.attn.proj.weight                   transformer.layers.0.0.to_out.0.weight
        blocks.0.attn.proj.bias                     transformer.layers.0.0.to_out.0.bias
                            -----------MLP Layer-------------
        blocks.0.norm2.weight                       transformer.layers.0.1.net.0.weight
        blocks.0.norm2.bias                         transformer.layers.0.1.net.0.bias
        blocks.0.mlp.fc1.weight                     transformer.layers.0.1.net.1.weight
        blocks.0.mlp.fc1.bias                       transformer.layers.0.1.net.1.bias
        blocks.0.mlp.fc2.weight                     transformer.layers.0.1.net.4.weight
        blocks.0.mlp.fc2.bias                       transformer.layers.0.1.net.4.bias
                .                                                      .
                .                                                      .
                .                                                      .
        """
        jax_dict = torch.load(pretain_path, map_location='cpu')
        new_dict = {}

        def add_item(key, value):
            key = key.replace('blocks', 'transformer.layers')
            new_dict[key] = value
            
        for key, value in jax_dict.items():
            if key == 'cls_token':
                new_dict[key] = value
            
            elif 'norm1' in key:
                new_key = key.replace('norm1', '0.norm')
                add_item(new_key, value)
            elif 'attn.qkv' in key:
                new_key = key.replace('attn.qkv', '0.to_qkv')
                add_item(new_key, value)
            elif 'attn.proj' in key:
                new_key = key.replace('attn.proj', '0.to_out.0')
                add_item(new_key, value)
            elif 'norm2' in key:
                new_key = key.replace('norm2', '1.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '1.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '1.net.4')
                add_item(new_key, value)
            elif 'patch_embed.proj' in key:
                new_key = key.replace('patch_embed.proj', 'to_patch_embedding.0')
                add_item(new_key, value)
            
            elif key == '   ':
                add_item('pos_embedding', value)
            elif key == 'norm.weight':
                add_item('transformer.norm.weight', value)
            elif key == 'norm.bias':
                add_item('transformer.norm.bias', value)
            
        self.load_state_dict(new_dict, strict=False)

    def forward(self, img):
        x = self.to_patch_embedding(img)  #img [B, C, H, W]  batch, 3, 224, 224
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape     # [B, N, C] x batch, 196(14*14), 768   通过一个线性层，14*14是分割后的小块的个数

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
 
        x, dots_list = self.transformer(x) # x [B, N+1, C], dots_list  1*batch*frame, 196
    
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x, dots_list



class VTN(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_frozen, spatial_size, spatial_args, temporal_type, temporal_args, spatial_suffix=''):
        super().__init__()
        self.frames = frames

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        # self.spatial_transformer = timm.create_model(f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', pretrained=True, **vars(spatial_args))
        
        pretrain_path = getattr(spatial_args, "pretrain_path", None)
        self.spatial_transformer = ViT(
          image_size = 224,
          patch_size = 16,
          dim = 768,
          depth = 12,
          heads = 12,
          mlp_dim = 3072,
          dropout = 0.1,
          emb_dropout = 0.1,
          pretrain_path =pretrain_path
)

        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
          self.spatial_transformer.eval()
          for param in self.spatial_transformer.parameters():
                param.requires_grad = False
        # Spatial preprocess
        self.preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(img_size),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          # transforms.Normalize(mean=self.spatial_transformer.default_cfg['mean'], std=self.spatial_transformer.default_cfg['std'])
        ])

        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.train_preprocess = create_transform(**config, is_training=True)

       
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)
        # self.spatial2temporal = Rearrange('b f d -> (b f) d')
        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames
        
        if temporal_type == 'longformer':
          self.temporal_transformer = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
          self.temporal_transformer = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
          self.temporal_transformer = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim * 2 ),
            nn.Linear(temporal_args.dim * 2, temporal_args.dim),
            nn.ReLU(),
            nn.Linear(temporal_args.dim, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)


        #additional
        self.tokensflatten = Rearrange('b f d -> b (f d)')
        self.mlp_addition = nn.Sequential(nn.Linear(temporal_args.dim * frames, int(temporal_args.dim * frames/2)), 
                                          nn.ReLU(), nn.Dropout(0.2), 
                                          nn.Linear(int(temporal_args.dim * frames/2), temporal_args.dim))
        
        # self.mlp_addition = nn.Sequential(nn.Linear(temporal_args.dim * frames, temporal_args.dim))
  

    def forward(self, img):
        #输入，batch * frame * channel * height * weight(4,16,3,224,224) 输出(batch * frame) * channel * height * weight
        x = self.collapse_frames(img)

        # Spatial Transformer
        if self.spatial_frozen:
          with torch.no_grad():
            x, dots_list = self.spatial_transformer(x)
        else:
          x, dots_list = self.spatial_transformer(x)
        
        dots_list = rearrange(dots_list, 'a (b f) d -> (a b) f d', f= self.frames, d = 196)
        #get head   (batch * frame) * 197(224^2 / 16^2 +1) * 768  (16*16*3,sequence length)
        x = x[:,0,:]
        
        # Spatial to temporal((batch * frame) * 768)  
        # 取了cls token后，解成batch * frame * 768
        x = self.spatial2temporal(x)

        #batch * (frame * 768)
        #addition
        tokens_flatten = self.tokensflatten(x)
        tokens_flatten = self.mlp_addition(tokens_flatten)

        # Temporal Transformer,输出 batch * 768
        x = self.temporal_transformer(x)
        x = torch.cat([x, tokens_flatten], 1)

        # Classifier
        return self.mlp_head(x), dots_list
    
    def forward_new(self, img):
        """
        用于 TensorFormer 特征提取：img [B, F, C, H, W] -> [B, F, D]
        """
        # 输入形状: [B, F, C, H, W] → [B*F, C, H, W]
        x = self.collapse_frames(img)

        # Spatial Transformer 输出（只保留 cls token 向量）
        if self.spatial_frozen:
            with torch.no_grad():
                x, _ = self.spatial_transformer(x)
        else:
            x, _ = self.spatial_transformer(x)

        # 取 cls token 向量 [B*F, D]
        x = x[:, 0, :]

        # 恢复时间维度: [B*F, D] -> [B, F, D]
        x = self.spatial2temporal(x)

        # Temporal Transformer
        x = self.temporal_transformer.forward_new(x)

        return x  # shape: [B, F, D]
"""
import torch
from argparse import Namespace

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置对象
config = Namespace(
    frames=60,
    num_classes=2,
    img_size=224,
    patch_size=16,
    spatial_frozen=True,
    spatial_size='base',
    spatial_args={'num_classes': 2},
    temporal_type='longformer',
    temporal_args={
        'seq_len': 60,
        'dim': 768,
        'depth': 3,
        'heads': 12,
        'dim_head': 64,
        'mlp_dim': 3072,
        "attention_window": 8,
        'dropout': 0.1
    }
)

# 初始化模型
model = VTN(**vars(config)).to(device=2)
model.eval()

# 模拟一个 batch 输入：[B, F, C, H, W]
dummy_input = torch.randn(2, 60, 3, 224, 224).to(device=2)

# 输出测试
with torch.no_grad():
    logits, dots = model(dummy_input)
    feat = model.forward_new(dummy_input)

print("分类输出 logits shape:", logits.shape)    # 应为 [2, num_classes]
print("特征输出 feat shape:", feat.shape)        # 应为 [2, 60, 768]
"""