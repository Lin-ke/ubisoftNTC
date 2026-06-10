import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _safe_font(size=12):
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _vis_psnr(pred, ref):
    mse = F.mse_loss(pred, ref).item()
    return -10 * np.log10(max(mse, 1e-10))


def _tensor_to_hwc(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def _to_rgb(img_np):
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    return img_np


def _add_label(img_np, label, psnr=None, font=None):
    h, w = img_np.shape[:2]
    bar_h = 18
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.float32)
    canvas[bar_h:, :] = img_np
    pil = Image.fromarray((canvas * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    text = f"{label}" if psnr is None else f"{label}  PSNR:{psnr:.2f}"
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.array(pil, dtype=np.float32) / 255.0


def _make_border(img_np, color, thickness=2):
    h, w = img_np.shape[:2]
    c = img_np.shape[2] if img_np.ndim == 3 else 1
    if c == 1:
        img_np = np.stack([img_np[:, :, 0]] * 3, axis=-1)
    bordered = np.ones((h + thickness * 2, w + thickness * 2, 3), dtype=np.float32)
    bordered[:, :] = color
    bordered[thickness:-thickness, thickness:-thickness] = img_np
    return bordered


_CHANNEL_SPECS = {
    'Albedo': {
        'pred': (0, 3), 'ref': (0, 3), 'color': (0.8, 0.2, 0.2),
    },
    'Normal': {
        'pred': (3, 5), 'ref': (3, 6), 'color': (0.2, 0.6, 0.8),
    },
    'AO': {
        'pred': (6, 7), 'ref': (6, 7), 'color': (0.6, 0.6, 0.6),
    },
    'Roughness': {
        'pred': (7, 8), 'ref': (7, 8), 'color': (0.4, 0.8, 0.4),
    },
    'Metalness': {
        'pred': (8, 9), 'ref': (8, 9), 'color': (0.9, 0.7, 0.2),
    },
}
_CHANNEL_ORDER = ['Albedo', 'Normal', 'AO', 'Roughness', 'Metalness']
_COMPACT_CHANNEL_SPECS = {
    'Albedo': {'pred': (0, 3), 'ref': (0, 3), 'color': (0.8, 0.2, 0.2)},
    'Normal': {'pred': (3, 5), 'ref': (3, 5), 'color': (0.2, 0.6, 0.8)},
    'AO':     {'pred': (5, 6), 'ref': (5, 6), 'color': (0.6, 0.6, 0.6)},
}
_CHANNEL_ALIASES = {
    'albedo': 'Albedo', 'basecolor': 'Albedo', 'base_color': 'Albedo',
    'normal': 'Normal', 'normalxy': 'Normal',
    'ao': 'AO', 'occlusion': 'AO',
    'rough': 'Roughness', 'roughness': 'Roughness',
    'metal': 'Metalness', 'metallic': 'Metalness', 'metalness': 'Metalness',
}


def _normalize_channel_names(channels):
    if channels is None:
        return None
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(',') if c.strip()]

    names = []
    for ch in channels:
        if isinstance(ch, dict):
            ch = ch.get('name') or ch.get('label')
        if ch is None:
            continue
        name = _CHANNEL_ALIASES.get(str(ch).lower(), str(ch))
        if name in _CHANNEL_SPECS and name not in names:
            names.append(name)
    return names


def _get_specs(ref_dim):
    return _COMPACT_CHANNEL_SPECS if ref_dim == 6 else _CHANNEL_SPECS


def _default_inference_channels(output_dim, ref_dim=9):
    names = []
    specs = _get_specs(ref_dim)
    for name in _CHANNEL_ORDER:
        if name not in specs:
            continue
        spec = specs[name]
        if output_dim >= spec['pred'][1] and ref_dim >= spec['ref'][1]:
            names.append(name)
    return names


def _channel_indices(names, ref_dim=9):
    idx = []
    specs = _get_specs(ref_dim)
    for name in names:
        if name not in specs:
            continue
        lo, hi = specs[name]['pred']
        idx.extend(range(lo, hi))
    return idx


@torch.no_grad()
def visualize_comparison(model, material_data, device, output_path, scale=0.0):
    model.eval()
    font = _safe_font()
    ref_tensor = material_data['ref_tensor'].to(device)
    name = material_data['name']
    h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    mip_level = int(scale)
    mipmaps = material_data.get('mipmaps', [])
    if 0 <= mip_level < len(mipmaps):
        ref_tensor = mipmaps[mip_level].to(device)
        h, w = ref_tensor.shape[1], ref_tensor.shape[2]

    u = torch.linspace(0, 1, w, device=device)
    v = torch.linspace(0, 1, h, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1).unsqueeze(0)
    scales = torch.tensor([float(scale)], device=device)
    pred = model(uv, scales)

    channels = _normalize_channel_names(material_data.get('channels'))
    if channels is None:
        if hasattr(model, 'get_inference_channels'):
            channels = model.get_inference_channels()
        else:
            channels = _default_inference_channels(pred.shape[1], ref_tensor.shape[0])

    def _pred_ref_for(label):
        spec = _get_specs(ref_tensor.shape[0])[label]
        plo, phi = spec['pred']
        rlo, rhi = spec['ref']
        pred_ch = pred[0, plo:phi]
        if label == 'Normal':
            from ntc_utils import reconstruct_normal
            pred_ch = reconstruct_normal(pred_ch.unsqueeze(0)).squeeze(0)
        return pred_ch, ref_tensor[rlo:rhi]

    rows = []
    specs = _get_specs(ref_tensor.shape[0])
    for label in channels:
        pred_ch, ref_ch = _pred_ref_for(label)
        psnr = _vis_psnr(pred_ch.unsqueeze(0), ref_ch.unsqueeze(0))
        pred_np = _to_rgb(_tensor_to_hwc(pred_ch))
        ref_np = _to_rgb(_tensor_to_hwc(ref_ch))
        diff = np.abs(pred_np - ref_np)

        panels = [pred_np, ref_np, diff]
        titles = [f'{label} Pred', f'{label} Ref', f'{label} Diff']

        labeled = []
        for idx, (p, t) in enumerate(zip(panels, titles)):
            p = _add_label(p, t, psnr if idx == 0 else None, font=font)
            p = _make_border(p, specs[label]['color'], thickness=2)
            labeled.append(p)

        min_h = min(p.shape[0] for p in labeled)
        labeled = [p[:min_h] for p in labeled]
        row = np.concatenate(labeled, axis=1)
        rows.append(row)

    if not rows:
        raise ValueError('No valid inference channels to visualize')

    min_w = min(r.shape[1] for r in rows)
    rows = [r[:, :min_w] for r in rows]
    sep = np.zeros((4, min_w, 3), dtype=np.float32)
    final_rows = []
    for i, r in enumerate(rows):
        final_rows.append(r)
        if i < len(rows) - 1:
            final_rows.append(sep)

    full = np.concatenate(final_rows, axis=0)
    title_h = 22
    title_bar = np.zeros((title_h, full.shape[1], 3), dtype=np.float32)
    title_bar[:, :] = (0.1, 0.1, 0.15)
    full = np.concatenate([title_bar, full], axis=0)

    pil = Image.fromarray((full * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    header = f"Material: {name}  |  Scale (mip): {scale:.1f}  |  Channels: {','.join(channels)}"
    draw.text((6, 3), header, fill=(255, 255, 255), font=font)
    pil.save(output_path)
