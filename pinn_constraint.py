import torch
import torch.nn as nn
from gridencoder import GridEncoder

class HashPINN(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 output_dim=1, 
                 num_layers=3, 
                 hidden_dim=64,
                 base_resolution=16,
                 max_resolution=2048,
                 num_levels=16,
                 level_dim=2,
                 log2_hashmap_size=19):
        """
        基于 Hash 编码的物理信息神经网络 (PINN)，用于约束 3D 高斯椭球的属性。
        """
        super().__init__()
        
        # 使用工作区中现有的 GridEncoder (Hash Encoding)
        self.encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=max_resolution
        )
        
        # PINN MLP 解码器部分
        in_dim = self.encoder.output_dim
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # 输出层（例如：可以是密度分布、SDF 或者物理场的约束值）
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        x: [N, 3] 空间坐标，通常为高斯的均值 (xyz)
        """
        # 注意: GridEncoder 通常默认输入在 [0, 1] 范围内或者特定边界内
        # 可以根据场景尺度进行归一化，如 x = (x - bbox_min) / (bbox_max - bbox_min)
        h = self.encoder(x)
        out = self.mlp(h)
        return out
        
    def compute_physics_loss(self, xyz, opacity, scales, rotations):
        """
        计算针对 3D Gaussian Splatting 的 PINN 物理约束损失。
        根据你的具体需求，这里可以计算如 Eikonal Loss 控制表面平滑，或流体/力学方程残差。
        """
        # 允许求导以计算空间梯度 (如计算偏微分方程残差PDE)
        xyz.requires_grad_(True)
        
        # 预测空间场的物理属性
        pred_field = self.forward(xyz)
        
        # --- 示例：基于梯度的物理约束 (Eikonal Loss 约束距离场) ---
        grad_outputs = torch.ones_like(pred_field)
        gradients = torch.autograd.grad(
            outputs=pred_field,
            inputs=xyz,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 例如：(|∇P| - 1)^2，迫使梯度模长为1 (SDF特性)
        eikonal_loss = ((gradients.norm(p=2, dim=-1) - 1.0) ** 2).mean()
        
        # --- 示例：几何约束 (结合高斯的 scale 或 opacity) ---
        # 假设 pred_field 预测的局部特征需要与高斯的尺度 (scale) 或不透明度匹配
        # scale_constraint = F.mse_loss(pred_field.squeeze(-1), torch.mean(scales, dim=-1))
        
        # 总损失
        total_pinn_loss = eikonal_loss # + lambda_s * scale_constraint
        
        return total_pinn_loss

