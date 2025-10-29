Environment Install:

pip install -r requirements.txt



Have Category Classification:
--> run scripts/prepare_split.py --src data_raw --dst data_split ...

Have images + labels.csv:
--> run scripts/prepare_split_from_csv.py ...


Train:
python scripts/train_baseline.py --data data_split --model mobilenetv3_small_100 --loss ce_ls --epochs 30 --out checkpoints/baseline.pt



Calibration & Unsure:
python scripts/calibrate_unsure.py --ckpt checkpoints/baseline.pt --data data_split --split val



Cross-domain Evaluation:
python scripts/eval_cross_dataset.py --ckpt checkpoints/baseline.pt --d1 data_split_wcd --d2 data_split_trashnet --split test



Run Gradio:
python app1.py



Evidence Pack:
python make_evidence_pack.py



Model Training:

MobileNetV3(CNN):
python scripts\train_baseline.py --data data_split --model mobilenetv3_small_100 --loss ce_ls --epochs 30 --out checkpoints\mbv3_small.pt

ViT/DeiT-Tiny(Parallel Modern Architecture Route)
python scripts\train_baseline.py --data data_split --model deit_tiny_patch16_224 --loss ce_ls --epochs 50 --out checkpoints\vit_tiny.pt

