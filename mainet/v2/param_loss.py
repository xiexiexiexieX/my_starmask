"""
param_loss.py — φ 方向 + 点/线门控辅助监督（仅 v2 使用）
=========================================================
"""
import torch
import torch.nn.functional as F


def _angle_to_sincos(a):
    return torch.stack([torch.sin(2 * a), torch.cos(2 * a)])


def param_supervision_loss(pred_params, gt_params_list):
    device = pred_params['phi'].device
    angle_total = torch.tensor(0.0, device=device)
    gate_total  = torch.tensor(0.0, device=device)
    angle_n, gate_n = 0, 0
    detail_phi = 0.0

    B = pred_params['phi'].shape[0]
    for b in range(B):
        gt = gt_params_list[b]
        if gt is None:
            continue
        if 'gate_target' in gt:
            l = F.binary_cross_entropy_with_logits(
                pred_params['gate_logit'][b], gt['gate_target'].to(device).view(1))
            gate_total = gate_total + l; gate_n += 1
        if 'phi' in gt:
            gt_sc = _angle_to_sincos(gt['phi'].to(device).float())
            l = F.smooth_l1_loss(pred_params['phi_sc'][b], gt_sc)
            angle_total = angle_total + l; angle_n += 1
            detail_phi += l.item()

    return {
        'angle': angle_total / max(angle_n, 1),
        'gate':  gate_total  / max(gate_n, 1),
        'angle_n': angle_n, 'gate_n': gate_n,
        'phi':   detail_phi / max(angle_n, 1) if angle_n > 0 else 0.0,
    }
