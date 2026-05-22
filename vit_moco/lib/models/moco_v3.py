"""
@来源: MoCo v3 builder.py（Attribution-NonCommercial 4.0）
       https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
@功能简述: MoCo v3 自监督框架 — 双编码器 + projector + predictor + 对比损失
@主要用途: 包装 ResNet / ViT backbone（如 vit_3d_base），供 MoCo3DTrainer 预训练

@前向流程（forward）:
    输入同一样本的两个增强视图 x1, x2
        → base_encoder + projector + predictor → q1, q2（在线，有梯度）
        → momentum_encoder + projector          → k1, k2（动量，no_grad + EMA）
        → loss = contrastive_loss(q1,k2) + contrastive_loss(q2,k1)

@类一览:
    | 类           | 作用 |
    |--------------|------|
    | MoCo         | 基类：双 encoder、EMA、contrastive_loss、forward |
    | MoCo_ResNet  | projector 接在 .fc 上 |
    | MoCo_ViT     | projector 接在 ViT .head 上（3D 预训练常用） |

@初始化（MoCo.__init__）:
    1. 双 encoder
       base_encoder = base_encoder(num_classes=mlp_dim)   # 工厂，如 vit_3d_base
       momentum_encoder = 同上结构；在线学习，动量出 key
    2. projector（MoCo_ViT / MoCo_ResNet 子类实现 _build_projector_and_predictor_mlps）
       删除原分类 head，换 3 层 MLP：embed_dim → mlp_dim → mlp_dim → dim
       例 ViT-Base：768 → 4096 → 4096 → 256；两个 encoder 各一套 projector
    3. predictor（仅 MoCo 模块 self.predictor，2 层 MLP：dim → mlp_dim → dim）
       只用于在线分支生成 q；momentum_encoder 无 predictor
    4. 动量网：参数拷贝自 base_encoder；requires_grad=False；训练中 EMA：
       θ_m ← m·θ_m + (1-m)·θ_b

@projector vs predictor:
    |          | 挂在哪                         | 两个 encoder 都有？ |
    |----------|--------------------------------|----------------------|
    | projector | encoder.head（3 层 MLP）      | 是                   |
    | predictor | self.predictor，仅包在线输出  | 否（仅在线 q 侧）    |

@contrastive_loss(q, k):
    1. L2 归一化 q、k
    2. 分布式：concat_all_gather(k) 扩大负样本（见 lib/utils/distributed.py）
    3. logits = q @ k.T / T
    4. CrossEntropy：每个 q 匹配对应正样本 k；异图为负样本

@对外暴露:
    __all__ = ["MoCo", "MoCo_ResNet", "MoCo_ViT"]
"""

import torch
import torch.nn as nn
from vit_moco.lib.utils.distributed import concat_all_gather

__all__ = ["MoCo", "MoCo_ResNet", "MoCo_ViT"]


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # check if distributed training is available and initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all targets
            k = concat_all_gather(k)
            N = q.shape[0]  # batch size per GPU
            labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).to(q.device)
        else:
            N = q.shape[0]  # batch size
            labels = torch.arange(N, dtype=torch.long).to(q.device)
        
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        # print("running_mean: ", self.base_encoder.patch_embed.proj[1].running_mean.mean())
        # print("running_var: ", self.base_encoder.patch_embed.proj[1].running_var.mean())
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)



