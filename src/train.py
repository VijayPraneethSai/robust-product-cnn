"""Full pipeline on real Fashion-MNIST product photos:

1. Train the VAE (adversary's generative strategy space) on clean images.
2. Train the CDL layer (feature selection over the VAE's latent space).
3. Train an undefended CNN classifier on clean data only.
4. Attack phase: run the CDL+SA Stackelberg-game adversary against the
   undefended CNN, measure how often it fools it.
5. Defense phase (the half the original repo never finished): retrain the
   CNN on a mix of clean data and freshly generated adversarial examples,
   for several game rounds, each round regenerating adversarial examples
   against the CURRENT classifier - this is the "retrain the learner over
   the adversarial manipulations" step the paper's method actually asks for.
6. Attack the DEFENDED classifier with a fresh, independent attack run and
   report honest before/after numbers, plus clean accuracy, so the defense
   is checked both ways: did it get robust, and did it stay accurate.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn

from data import load_fashion_mnist, CLASSES
from models import VAE, vae_loss, CDLLayer, ProductCNN
from attack import sa_attack

torch.manual_seed(0)
np.random.seed(0)


def batches(X, y, bs, shuffle=True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield torch.tensor(X[b]), torch.tensor(y[b])


def train_vae(Xtr, epochs=6, bs=128):
    vae = VAE(latent_dim=32)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    n = len(Xtr)
    for ep in range(epochs):
        tot = 0.0
        for xb, _ in batches(Xtr, np.zeros(n), bs):
            opt.zero_grad()
            recon, mu, logvar, z = vae(xb)
            loss = vae_loss(recon, xb, mu, logvar) / xb.size(0)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"  VAE epoch {ep+1}/{epochs}  loss={tot/n:.2f}")
    return vae


def train_cdl(vae, Xtr, epochs=8, bs=256):
    """Zhu et al. 2016's objective: reconstruction + tau * L2,p-norm group
    sparsity on the analysis dictionary V. Their paper's exact optimizer is
    an iterative reweighted least squares (IRLS) algorithm with a proven
    fixed-point convergence guarantee; I couldn't retrieve that pseudocode
    (the AAAI proceedings PDF is behind a bot-blocker I can't get past), so
    this uses ordinary gradient descent (Adam) on the same objective
    instead - a real, acknowledged substitution, not a claim of using IRLS."""
    cdl = CDLLayer(latent_dim=32, dict_size=64, tau=0.01, p=0.7)
    opt = torch.optim.Adam(cdl.parameters(), lr=1e-3)
    n = len(Xtr)
    for ep in range(epochs):
        tot, tot_recon, tot_sparse = 0.0, 0.0, 0.0
        for xb, _ in batches(Xtr, np.zeros(n), bs):
            with torch.no_grad():
                mu, _ = vae.encode(xb)
            opt.zero_grad()
            loss, recon_loss, sparsity_loss = cdl.cdl_loss(mu)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            tot_recon += recon_loss.item() * xb.size(0)
            tot_sparse += sparsity_loss.item() * xb.size(0)
        print(f"  CDL epoch {ep+1}/{epochs}  loss={tot/n:.4f}  recon={tot_recon/n:.4f}  "
              f"sparsity={tot_sparse/n:.2f}")
    importance = cdl.feature_importance()
    print(f"  feature importance (V row norms): min={importance.min():.3f} max={importance.max():.3f} "
          f"— top 10 dims: {torch.topk(importance, 10).indices.tolist()}")
    return cdl


def train_classifier(Xtr, ytr, Xval, yval, epochs=5, bs=128, model=None, extra_X=None, extra_y=None):
    clf = model if model is not None else ProductCNN()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        clf.train()
        tot = 0.0
        for xb, yb in batches(Xtr, ytr, bs):
            if extra_X is not None and len(extra_X) > 0:
                # mix in adversarial examples from the current game round
                k = min(bs // 2, len(extra_X))
                sel = np.random.choice(len(extra_X), k, replace=False)
                xb = torch.cat([xb, extra_X[sel]], dim=0)
                yb = torch.cat([yb, extra_y[sel]], dim=0)
            opt.zero_grad()
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        val_acc = evaluate(clf, Xval, yval)
        print(f"  clf epoch {ep+1}/{epochs}  loss={tot/n:.3f}  val_acc={val_acc:.4f}")
    return clf


@torch.no_grad()
def evaluate(clf, X, y, bs=256):
    clf.eval()
    correct = 0
    for xb, yb in batches(torch.tensor(X) if not torch.is_tensor(X) else X,
                           torch.tensor(y) if not torch.is_tensor(y) else y, bs, shuffle=False):
        pred = clf(xb).argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / len(X)


def run_attack(vae, cdl, clf, X, y, n_dims, n_iters, budget, bs=128):
    fooled_total, n_total = 0, 0
    adv_images, adv_labels = [], []
    for xb, yb in batches(torch.tensor(X), torch.tensor(y), bs, shuffle=False):
        adv_img, fooled, pred = sa_attack(vae, cdl, clf, xb, yb, n_dims=n_dims, n_iters=n_iters, budget=budget)
        fooled_total += fooled.sum().item()
        n_total += xb.size(0)
        adv_images.append(adv_img)
        adv_labels.append(yb)
    attack_success_rate = fooled_total / n_total
    return attack_success_rate, torch.cat(adv_images), torch.cat(adv_labels)


import os


def main():
    t0 = time.time()
    Xtr_full, ytr_full, Xte, yte = load_fashion_mnist()
    # held-out validation slice of the training set (never used for the
    # attack/defense evaluation numbers, only for picking when to stop)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr_full))
    val_n = 6000
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    Xtr, ytr = Xtr_full[train_idx], ytr_full[train_idx]
    Xval, yval = Xtr_full[val_idx], ytr_full[val_idx]
    print(f"train={len(Xtr)}  val={len(Xval)}  test={len(Xte)} (real Fashion-MNIST, Zalando product photos)")

    if os.path.exists("results/vae.pt"):
        print("found results/vae.pt - resuming, skipping VAE training")
        vae = VAE(latent_dim=32)
        vae.load_state_dict(torch.load("results/vae.pt"))
    else:
        print("training VAE (adversary's generative strategy space)...")
        vae = train_vae(Xtr, epochs=18)
        torch.save(vae.state_dict(), "results/vae.pt")

    if os.path.exists("results/cdl.pt"):
        print("found results/cdl.pt - resuming, skipping CDL training")
        cdl = CDLLayer(latent_dim=32, dict_size=64, tau=0.01, p=0.7)
        cdl.load_state_dict(torch.load("results/cdl.pt"))
    else:
        print("training CDL layer (feature selection over the latent space)...")
        cdl = train_cdl(vae, Xtr, epochs=15)
        torch.save(cdl.state_dict(), "results/cdl.pt")

    if os.path.exists("results/clf_undefended.pt"):
        print("found results/clf_undefended.pt - resuming, skipping undefended classifier training")
        clf_undefended = ProductCNN()
        clf_undefended.load_state_dict(torch.load("results/clf_undefended.pt"))
    else:
        print("training undefended classifier on clean data only...")
        clf_undefended = train_classifier(Xtr, ytr, Xval, yval, epochs=15)
        torch.save(clf_undefended.state_dict(), "results/clf_undefended.pt")

    clean_acc_undefended = evaluate(clf_undefended, Xte, yte)
    print(f"undefended clean test accuracy: {clean_acc_undefended:.4f}")

    # attack subset: use a few hundred test images for the SA search (it's
    # the expensive step - this is the realistic "evaluate against a held-out
    # attack set" setting, not the full 10k)
    n_attack = 600
    att_idx = rng.choice(len(Xte), n_attack, replace=False)
    Xatt, yatt = Xte[att_idx], yte[att_idx]

    if os.path.exists("results/checkpoint.json"):
        ckpt = json.load(open("results/checkpoint.json"))
    else:
        ckpt = {}

    if "undefended_attack_success_rate" in ckpt:
        asr_undefended = ckpt["undefended_attack_success_rate"]
        print(f"found cached undefended attack success rate: {asr_undefended:.4f} - resuming")
    else:
        print(f"attacking the undefended classifier with the CDL+SA Stackelberg adversary "
              f"({n_attack} test images)...")
        asr_undefended, adv_imgs_1, adv_labels_1 = run_attack(
            vae, cdl, clf_undefended, Xatt, yatt, n_dims=10, n_iters=40, budget=2.5
        )
        print(f"undefended attack success rate: {asr_undefended:.4f}")
        ckpt["undefended_attack_success_rate"] = asr_undefended
        ckpt["undefended_clean_test_acc"] = clean_acc_undefended
        json.dump(ckpt, open("results/checkpoint.json", "w"))

    # --- defense: the Stackelberg retraining loop the original repo never
    # finished --- generate adversarial examples against the CURRENT
    # classifier each round, retrain on clean+adversarial, repeat. Uses a
    # FIXED pool of images re-attacked every round (not a fresh random
    # subset each time) - an earlier version of this project tried fresh
    # subsets each round and the defense never converged, bouncing back up
    # to the undefended attack rate; forcing the model to repeatedly face
    # down the SAME images as they evolve round to round is what standard
    # adversarial training actually does, and is the fix for that.
    print("running Stackelberg defense rounds (fixed adversarial pool, "
          "re-attacked against the current classifier every round)...")
    n_rounds = 6
    pool_n = 2500
    # the pool must be the SAME across restarts, so seed it independently of
    # rng's position (which depends on how much was already consumed above)
    pool_rng = np.random.default_rng(42)
    pool_idx = pool_rng.choice(len(Xtr), pool_n, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]

    round_history = ckpt.get("round_history", [])
    start_round = len(round_history)
    clf_defended = ProductCNN()
    if start_round > 0 and os.path.exists("results/clf_defended_latest.pt"):
        print(f"found {start_round} completed defense round(s) - resuming from round {start_round + 1}")
        clf_defended.load_state_dict(torch.load("results/clf_defended_latest.pt"))
    else:
        clf_defended.load_state_dict(clf_undefended.state_dict())  # start from the trained model

    for r in range(start_round, n_rounds):
        t_round = time.time()
        _, adv_imgs, adv_labels = run_attack(vae, cdl, clf_defended, Xpool, ypool,
                                              n_dims=10, n_iters=None, budget=None)
        clf_defended = train_classifier(Xtr, ytr, Xval, yval, epochs=3, model=clf_defended,
                                         extra_X=adv_imgs.detach(), extra_y=adv_labels.detach())
        round_val_acc = evaluate(clf_defended, Xval, yval)
        round_asr, _, _ = run_attack(vae, cdl, clf_defended, Xatt, yatt, n_dims=10, n_iters=None, budget=None)
        round_history.append({"round": r + 1, "val_acc": round_val_acc, "attack_success_rate": round_asr,
                               "round_time_sec": round(time.time() - t_round, 1)})
        print(f"round {r+1}/{n_rounds}: val_acc={round_val_acc:.4f}  attack_success_rate={round_asr:.4f}  "
              f"({time.time()-t_round:.0f}s)")
        # checkpoint after EVERY round: model weights + round history, so a
        # restart resumes from here instead of redoing already-done rounds
        torch.save(clf_defended.state_dict(), "results/clf_defended_latest.pt")
        ckpt["round_history"] = round_history
        json.dump(ckpt, open("results/checkpoint.json", "w"))

    clean_acc_defended = evaluate(clf_defended, Xte, yte)
    print(f"defended clean test accuracy: {clean_acc_defended:.4f}")

    print("attacking the DEFENDED classifier with a fresh, independent attack run...")
    asr_defended, _, _ = run_attack(vae, cdl, clf_defended, Xatt, yatt, n_dims=10, n_iters=40, budget=2.5)
    print(f"defended attack success rate: {asr_defended:.4f}")

    total_time = time.time() - t0
    results = {
        "dataset": "Fashion-MNIST (real, Zalando product photos, 10 apparel categories)",
        "train_n": len(Xtr), "val_n": len(Xval), "test_n": len(Xte), "attack_set_n": n_attack,
        "total_time_sec": round(total_time, 1),
        "undefended_clean_test_acc": clean_acc_undefended,
        "undefended_attack_success_rate": asr_undefended,
        "defense_rounds": round_history,
        "defended_clean_test_acc": clean_acc_defended,
        "defended_attack_success_rate": asr_defended,
    }
    with open("results/robust_cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(clf_undefended.state_dict(), "results/clf_undefended.pt")
    torch.save(clf_defended.state_dict(), "results/clf_defended.pt")
    print(json.dumps(results, indent=2))
    print("saved results/robust_cnn_results.json")


if __name__ == "__main__":
    main()
