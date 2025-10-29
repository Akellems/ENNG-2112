
# ─────────────────────────────────────────────────────────────
# File: scripts/export_onnx_and_quantize.py
# Purpose: Export ONNX and (optional) dynamic quantized PyTorch model
# Output: exports/*.onnx (if onnx installed), checkpoints/*_int8.pt
# ─────────────────────────────────────────────────────────────

import argparse, os
from pathlib import Path
import torch, timm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--onnx', action='store_true', help='also export ONNX if available')
    ap.add_argument('--img', type=int, default=224)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    model = timm.create_model(ck['model'], pretrained=False, num_classes=len(ck['classes']))
    model.load_state_dict(ck['state_dict']); model.eval()

    # dynamic quantization (Linear-dominated nets benefit; CNN less so, but small win)
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    qpath = Path(args.ckpt).with_suffix('').as_posix() + '_int8.pt'
    torch.save({'model': ck['model'], 'state_dict': qmodel.state_dict(), 'classes': ck['classes']}, qpath)

    if args.onnx:
        try:
            import torch.onnx
            Path('exports').mkdir(exist_ok=True)
            dummy = torch.randn(1,3,args.img,args.img)
            onnx_path = os.path.join('exports', Path(args.ckpt).stem + '.onnx')
            torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['logits'],
                              dynamic_axes={'input':{0:'batch'}, 'logits':{0:'batch'}}, opset_version=17)
            print('Exported ONNX:', onnx_path)
        except Exception as e:
            print('ONNX export failed:', e)

if __name__ == '__main__':
    main()

