import argparse
import datetime
import json
import pathlib

import dac
import torch
from torch.utils.data import DataLoader

from datasets.libri2mix_ds import Libri2MixDataset
from losses.pit_latent_mse import pit_mse
from models.separator_rwkv6 import RWKV6Separator
from utils.collate import collate_latent_batch
from utils.lr_sched import CosineWarmup
from utils.seed import seed_everything

# ------------------------------------------------------------
# command‑line helpers
# ------------------------------------------------------------

def get_cfg(path: str):
    """Load YAML/JSON but stay YAML‑agnostic to avoid extra dep."""
    import yaml

    with open(path) as fi:
        return yaml.safe_load(fi)


# ------------------------------------------------------------
# main training routine
# ------------------------------------------------------------

def main(cfg):
    # 0) deterministic reproducibility
    seed_everything(cfg.get("seed", 42))

    # 1) dataset & dataloader
    train_ds = Libri2MixDataset(
        cfg["train_csv"],
        segment=cfg["segment_sec"],
        cache_latents=True,
    )
    dev_ds = Libri2MixDataset(
        cfg["dev_csv"], segment=cfg["segment_sec"], cache_latents=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=collate_latent_batch,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_latent_batch,
    )

    # 2) model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = dac.DAC.load(dac.utils.download("16khz")).to(device).eval()
    model = RWKV6Separator(codec, depth=cfg["rwkv_depth"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    total_steps = cfg["epochs"] * len(train_loader) // cfg["grad_accum"]
    sched = CosineWarmup(
        opt,
        warmup=cfg["warmup"],
        max_steps=total_steps,
        min_lr=cfg.get("min_lr", 1e-5),
    )

    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"])

    # 3) logging dirs
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = pathlib.Path("runs") / run_name
    tb = torch.utils.tensorboard.SummaryWriter(log_root)
    ckpt_dir = pathlib.Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_sisdr = -1e9

    for epoch in range(cfg["epochs"]):
        model.train()
        for b, (mix, lat_paths) in enumerate(train_loader):
            mix = mix.to(device).unsqueeze(1)  # [B,1,T]
            # flatten list of tuples: [(s1,s2), ...] → [s1,s2,...]
            flat_paths = sum(lat_paths, ())
            tgt_lat = [torch.load(p)["z"].to(device) for p in flat_paths]

            with torch.amp.autocast(device.type, enabled=cfg["amp"]):
                pred_lat, _ = model(mix)
                loss = pit_mse(pred_lat, tgt_lat[0::2], tgt_lat[1::2])
                loss = loss / cfg["grad_accum"]

            scaler.scale(loss).backward()

            if (b + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                tb.add_scalar("train/loss", loss.item() * cfg["grad_accum"], step)

        # dev evaluation per epoch
        sisdr = evaluate(model, dev_loader, device)
        tb.add_scalar("dev/SI-SDR", sisdr, step)
        if sisdr > best_sisdr:
            best_sisdr = sisdr
            save_ckpt(model, ckpt_dir / "best.pt", step, epoch)
        save_ckpt(model, ckpt_dir / f"epoch{epoch:02d}.pt", step, epoch)
        print(f"Epoch {epoch} done | dev SI-SDR {sisdr:.2f} | best {best_sisdr:.2f}")


# ------------------------------------------------------------
# helper functions
# ------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    tot = 0.0
    cnt = 0
    with torch.no_grad():
        for mix, lat_paths in loader:
            mix = mix.to(device).unsqueeze(1)
            tgt1, tgt2 = [
                torch.load(p)["z"].to(device) for p in sum(lat_paths, ())
            ]
            pred, _ = model(mix)
            loss = pit_mse(pred, tgt1, tgt2, reduce=False)
            sisdr = -10 * torch.log10(loss.mean()).item()
            tot += sisdr
            cnt += 1
    model.train()
    return tot / cnt


def save_ckpt(model, path, step, epoch):
    torch.save({"model": model.state_dict(), "step": step, "epoch": epoch}, path)


# ------------------------------------------------------------
# entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/train_thesis_model.yaml")
    args = parser.parse_args()

    cfg = get_cfg(args.cfg)
    main(cfg)

