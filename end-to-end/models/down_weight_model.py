import torch

ckpt_path = r"D:\USTH\nlp\NLP_Emotion_Group_14\end-to-end\results\electra+bert\checkpoints\bert_stage2_emotion.pth" 
KEEP_KEYS = {
    "epoch",
    "model_name",
    "pretrained_name",
    "num_labels",
    "tier_indices",
    "model_state",
    "val_macro_f1",
    "val_metrics",
    "threshold",
}

ckpt = torch.load(ckpt_path, map_location="cpu")
new_ckpt = {k: v for k, v in ckpt.items() if k in KEEP_KEYS}

if "model_state" in new_ckpt:
    new_ckpt["model_state"] = {
        k: v.half() if torch.is_floating_point(v) else v
        for k, v in new_ckpt["model_state"].items()
    }

torch.save(new_ckpt, ckpt_path)

print("✅ Done. Cleaned checkpoint saved.")
