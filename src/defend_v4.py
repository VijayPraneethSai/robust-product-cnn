"""v3 fixed the data-coverage problem (full dataset, not a 2,500-image
subset) but hit a textbook case of gradient masking / obfuscated gradients
(Athalye, Carlini & Wagner, "Obfuscated Gradients Give a False Sense of
Security," ICML 2018): a classifier trained only against a differentiable
gradient attack can become robust to THAT attack's gradient signal
specifically, without becoming genuinely robust - a black-box, gradient-free
attack (the original Stackelberg SA search) sailed straight through
(99.7% success, worse than before any defense at all).

The fix, consistent with that paper's own recommendation: train against
more than one kind of attack so the classifier can't just mask gradients
against a single one. Every epoch now mixes:
  - the fast PGD-style latent attack, on the FULL training set (keeps v3's
    fix for the coverage problem)
  - the real, slow, black-box Stackelberg search, on a smaller subset each
    epoch (expensive, but exposes the classifier to an attack that isn't
    fooled by gradient masking)
Final evaluation is still the full, independent, faithful Stackelberg
search - the only number that was ever the real test.
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
from defend_v3 import run_pgd_attack_eval, run_stackelberg_eval

torch.manual_seed(3)
np.random.seed(3)


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
    n_epochs = 6
    sb_subset_n = 1000  # per epoch, images attacked with the real slow Stackelberg search

    if os.path.exists("results/v4_ckpt.json"):
        ck = json.load(open("results/v4_ckpt.json"))
        start_epoch = ck["next_epoch"]
        model.load_state_dict(torch.load("results/v4_model_latest.pt"))
        print(f"found v4 checkpoint - resuming from epoch {start_epoch + 1}/{n_epochs}")
    else:
        # start from the v3 (gradient-attack-only) model: it's already
        # ahead on clean accuracy and on the attack it knows about, so this
        # continues training rather than throwing that away
        model.load_state_dict(torch.load("results/clf_defended_v3.pt"))
        print("starting ensemble (PGD + black-box Stackelberg) adversarial training from the v3 model")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_history = ck["epoch_history"] if start_epoch > 0 else []

    bs = 128
    for epoch in range(start_epoch, n_epochs):
        t_epoch = time.time()
        model.train()

        # the real black-box attack's examples for this epoch - generated
        # ONCE against the model at the START of the epoch (expensive),
        # then used throughout it; a fresh subset each epoch so the model
        # doesn't just memorize one fixed set
        sb_idx = rng.choice(len(Xtr), sb_subset_n, replace=False)
        Xsb, ysb = Xtr[sb_idx], ytr[sb_idx]
        t_sb = time.time()
        _, sb_adv_imgs, sb_adv_labels = [None, [], None]
        sb_imgs_list, sb_labels_list = [], []
        for xb, yb in batches(Xsb, ysb, 250, shuffle=False):
            adv_img, _, _ = stackelberg_attack(vae, cdl, model, xb, yb, n_dims=10)
            sb_imgs_list.append(adv_img)
            sb_labels_list.append(yb)
        Xsb_adv = torch.cat(sb_imgs_list, dim=0).detach()
        ysb_adv = torch.cat(sb_labels_list, dim=0)
        print(f"  generated {sb_subset_n} black-box Stackelberg adversarial examples "
              f"({time.time()-t_sb:.0f}s)")

        tot_loss, n_seen = 0.0, 0
        for xb, yb in batches(Xtr, ytr, bs, shuffle=True):
            adv_xb, _, _ = pgd_latent_attack(vae, model, xb, yb, n_steps=7, step_size=0.4, budget=2.5)
            full_x = torch.cat([xb, adv_xb], dim=0)
            full_y = torch.cat([yb, yb], dim=0)
            opt.zero_grad()
            logits = model(full_x)
            loss = loss_fn(logits, full_y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        # fold in the black-box Stackelberg examples, genuinely mixed with
        # clean data this time (a first version of this loop trained on
        # ONLY the 1,000 adversarial images for 3 full passes with nothing
        # to anchor normal classification, and the model collapsed to 13%
        # validation accuracy on epoch 1 - caught before any more time was
        # spent on it). One pass, each batch half clean / half adversarial,
        # same recipe as the PGD phase above.
        clean_idx_for_sb = rng.choice(len(Xtr), sb_subset_n, replace=False)
        Xclean_for_sb, yclean_for_sb = Xtr[clean_idx_for_sb], ytr[clean_idx_for_sb]
        for (xb_c, yb_c), (xb_a, yb_a) in zip(
            batches(Xclean_for_sb, yclean_for_sb, bs // 2, shuffle=True),
            batches(Xsb_adv.numpy(), ysb_adv.numpy(), bs // 2, shuffle=True),
        ):
            full_x = torch.cat([xb_c, torch.tensor(xb_a)], dim=0)
            full_y = torch.cat([yb_c, torch.tensor(yb_a)], dim=0)
            opt.zero_grad()
            logits = model(full_x)
            loss = loss_fn(logits, full_y)
            loss.backward()
            opt.step()

        val_acc = evaluate(model, Xval, yval)
        pgd_asr = run_pgd_attack_eval(vae, model, Xatt, yatt)
        epoch_time = time.time() - t_epoch
        epoch_history.append({"epoch": epoch + 1, "train_loss": tot_loss / n_seen, "val_acc": val_acc,
                               "pgd_attack_success_rate": pgd_asr, "epoch_time_sec": round(epoch_time, 1)})
        print(f"epoch {epoch+1}/{n_epochs}  loss={tot_loss/n_seen:.4f}  val_acc={val_acc:.4f}  "
              f"pgd_asr={pgd_asr:.4f}  ({epoch_time:.0f}s)")

        torch.save(model.state_dict(), "results/v4_model_latest.pt")
        json.dump({"next_epoch": epoch + 1, "epoch_history": epoch_history}, open("results/v4_ckpt.json", "w"))

    clean_acc = evaluate(model, Xte, yte)
    print(f"final clean test accuracy: {clean_acc:.4f}")

    print("evaluating against the SLOW, FAITHFUL, black-box Stackelberg attack "
          "(independent of training examples) ...")
    stackelberg_asr = run_stackelberg_eval(vae, cdl, model, Xatt, yatt)
    print(f"final Stackelberg attack success rate: {stackelberg_asr:.4f}")

    total_time = time.time() - t0
    results = {
        "method": "ensemble adversarial training: fast PGD-style latent attack on the full "
                  "dataset every epoch + the real black-box Stackelberg search on a fresh "
                  "1,000-image subset every epoch, fixing the gradient-masking failure of v3",
        "total_time_sec": round(total_time, 1),
        "epoch_history": epoch_history,
        "final_clean_test_acc": clean_acc,
        "final_pgd_attack_success_rate": epoch_history[-1]["pgd_attack_success_rate"],
        "final_stackelberg_attack_success_rate": stackelberg_asr,
        "undefended_clean_test_acc": 0.9194,
        "undefended_stackelberg_attack_success_rate": 0.84,
        "v3_clean_test_acc": 0.9197,
        "v3_stackelberg_attack_success_rate": 0.9967,
    }
    with open("results/robust_cnn_v4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), "results/clf_defended_v4.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
