"""Adversarial training at full-dataset scale, fixing the actual bug in the
first defense attempt: it only ever attacked a 2,500-image subset per round
(4.6% of the 54,000-image training set) because the faithful black-box
Stackelberg search is too slow to run on everything. This generates
adversarial examples with the fast, differentiable PGD-style latent attack
(attack.py's pgd_latent_attack) on EVERY training batch, every epoch -
white-box, used only for training - and then evaluates the result against
the original slow, faithful, black-box Stackelberg search, which is the
real, harder test and the one that counts.

Resumable: checkpoints after every epoch.
"""
import json
import os
import time

import numpy as np
import torch

from data import load_fashion_mnist
from models import VAE, CDLLayer, ProductCNN
from attack import pgd_latent_attack, stackelberg_attack
from train import batches, evaluate

torch.manual_seed(2)
np.random.seed(2)


def run_pgd_attack_eval(vae, classifier, X, y, bs=256):
    fooled_total, n_total = 0, 0
    for xb, yb in batches(torch.tensor(X), torch.tensor(y), bs, shuffle=False):
        _, fooled, _ = pgd_latent_attack(vae, classifier, xb, yb)
        fooled_total += fooled.sum().item()
        n_total += xb.size(0)
    return fooled_total / n_total


def run_stackelberg_eval(vae, cdl, classifier, X, y, bs=250):
    fooled_total, n_total = 0, 0
    for xb, yb in batches(torch.tensor(X), torch.tensor(y), bs, shuffle=False):
        _, fooled, _ = stackelberg_attack(vae, cdl, classifier, xb, yb, n_dims=10)
        fooled_total += fooled.sum().item()
        n_total += xb.size(0)
    return fooled_total / n_total


def main():
    t0 = time.time()
    Xtr_full, ytr_full, Xte, yte = load_fashion_mnist()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr_full))
    val_n = 6000
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    Xtr, ytr = Xtr_full[train_idx], ytr_full[train_idx]
    Xval, yval = Xtr_full[val_idx], ytr_full[val_idx]

    vae = VAE(latent_dim=32)
    vae.load_state_dict(torch.load("results/vae.pt"))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    cdl = CDLLayer(latent_dim=32, dict_size=64, tau=0.01, p=0.7)
    cdl.load_state_dict(torch.load("results/cdl.pt"))

    n_attack = 600
    att_idx = rng.choice(len(Xte), n_attack, replace=False)
    Xatt, yatt = Xte[att_idx], yte[att_idx]

    model = ProductCNN()
    start_epoch = 0
    n_epochs = 10

    if os.path.exists("results/v3_ckpt.json"):
        ck = json.load(open("results/v3_ckpt.json"))
        start_epoch = ck["next_epoch"]
        model.load_state_dict(torch.load("results/v3_model_latest.pt"))
        print(f"found v3 checkpoint - resuming from epoch {start_epoch + 1}/{n_epochs}")
    else:
        model.load_state_dict(torch.load("results/clf_undefended.pt"))
        print("starting full-dataset adversarial training from the undefended classifier")

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_history = ck["epoch_history"] if start_epoch > 0 else []

    bs = 128
    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()
        model.train()
        tot_loss, n_seen = 0.0, 0
        for xb, yb in batches(Xtr, ytr, bs, shuffle=True):
            adv_xb, _, _ = pgd_latent_attack(vae, model, xb, yb, n_steps=7, step_size=0.4, budget=2.5)
            # train on BOTH the clean batch and its adversarial version every
            # step (not a 50/50 sample-level mix) - full dataset coverage on
            # both clean and adversarial every single epoch
            full_x = torch.cat([xb, adv_xb], dim=0)
            full_y = torch.cat([yb, yb], dim=0)
            opt.zero_grad()
            logits = model(full_x)
            loss = loss_fn(logits, full_y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        val_acc = evaluate(model, Xval, yval)
        pgd_asr = run_pgd_attack_eval(vae, model, Xatt, yatt)
        epoch_time = time.time() - t_epoch
        epoch_history.append({"epoch": epoch + 1, "train_loss": tot_loss / n_seen, "val_acc": val_acc,
                               "pgd_attack_success_rate": pgd_asr, "epoch_time_sec": round(epoch_time, 1)})
        print(f"epoch {epoch+1}/{n_epochs}  loss={tot_loss/n_seen:.4f}  val_acc={val_acc:.4f}  "
              f"pgd_asr={pgd_asr:.4f}  ({epoch_time:.0f}s)")

        torch.save(model.state_dict(), "results/v3_model_latest.pt")
        json.dump({"next_epoch": epoch + 1, "epoch_history": epoch_history}, open("results/v3_ckpt.json", "w"))

    clean_acc = evaluate(model, Xte, yte)
    print(f"final clean test accuracy: {clean_acc:.4f}")

    print("evaluating against the SLOW, FAITHFUL, black-box Stackelberg attack "
          "(the real test) ...")
    stackelberg_asr = run_stackelberg_eval(vae, cdl, model, Xatt, yatt)
    print(f"final Stackelberg attack success rate: {stackelberg_asr:.4f}")

    total_time = time.time() - t0
    results = {
        "method": "full-dataset adversarial training with a fast differentiable PGD-style "
                  "latent attack (white-box, training-time only), evaluated against the slow "
                  "faithful black-box Stackelberg search",
        "total_time_sec": round(total_time, 1),
        "epoch_history": epoch_history,
        "final_clean_test_acc": clean_acc,
        "final_pgd_attack_success_rate": epoch_history[-1]["pgd_attack_success_rate"],
        "final_stackelberg_attack_success_rate": stackelberg_asr,
        "undefended_clean_test_acc": 0.9194,
        "undefended_stackelberg_attack_success_rate": 0.84,
    }
    with open("results/robust_cnn_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), "results/clf_defended_v3.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
