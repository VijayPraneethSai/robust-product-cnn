"""The first defense pass barely worked (71.8% attack success after
retraining vs 72.2% before - essentially unchanged). Root cause: each of
the 3 rounds generated adversarial examples for a *fresh random* 1500-image
subset, so the classifier never saw a repeated, evolving adversarial
version of the same images - standard adversarial training instead
re-attacks the SAME pool of images every round, so the model is forced to
actually close off the specific adversarial directions found against it,
round after round. This reruns the defense that way, with more rounds, a
larger fixed adversarial pool, and the full-strength attack (matching the
evaluation attack's search budget, not a cheaper one) used to generate
training-time adversarial examples too."""
import json
import time

import numpy as np
import torch

from data import load_fashion_mnist
from models import VAE, CDLLayer, ProductCNN
from attack import sa_attack
from train import train_vae, train_cdl, train_classifier, evaluate, run_attack, batches

torch.manual_seed(1)
np.random.seed(1)


def main():
    t0 = time.time()
    Xtr_full, ytr_full, Xte, yte = load_fashion_mnist()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr_full))
    val_n = 6000
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    Xtr, ytr = Xtr_full[train_idx], ytr_full[train_idx]
    Xval, yval = Xtr_full[val_idx], ytr_full[val_idx]

    print("re-training VAE + CDL (not saved from the first run)...")
    vae = train_vae(Xtr, epochs=6)
    cdl = train_cdl(vae, Xtr, epochs=3)
    torch.save(vae.state_dict(), "results/vae.pt")
    torch.save(cdl.state_dict(), "results/cdl.pt")

    clf_undefended = ProductCNN()
    clf_undefended.load_state_dict(torch.load("results/clf_undefended.pt"))
    clean_acc_undefended = evaluate(clf_undefended, Xte, yte)
    print(f"loaded undefended classifier, clean test acc={clean_acc_undefended:.4f}")

    n_attack = 500
    att_idx = rng.choice(len(Xte), n_attack, replace=False)
    Xatt, yatt = Xte[att_idx], yte[att_idx]
    asr_undefended, _, _ = run_attack(vae, cdl, clf_undefended, Xatt, yatt, n_dims=10, n_iters=40, budget=2.5)
    print(f"undefended attack success rate (recomputed, same as before): {asr_undefended:.4f}")

    print("stronger defense: fixed adversarial pool, re-attacked every round, "
          "full-strength attack settings...")
    clf_defended = ProductCNN()
    clf_defended.load_state_dict(clf_undefended.state_dict())

    pool_n = 4000
    pool_idx = rng.choice(len(Xtr), pool_n, replace=False)
    Xpool, ypool = Xtr[pool_idx], ytr[pool_idx]

    n_rounds = 6
    round_history = []
    for r in range(n_rounds):
        t_round = time.time()
        _, adv_imgs, adv_labels = run_attack(vae, cdl, clf_defended, Xpool, ypool,
                                              n_dims=10, n_iters=40, budget=2.5, bs=250)
        clf_defended = train_classifier(Xtr, ytr, Xval, yval, epochs=3, model=clf_defended,
                                         extra_X=adv_imgs.detach(), extra_y=adv_labels.detach())
        round_val_acc = evaluate(clf_defended, Xval, yval)
        round_asr, _, _ = run_attack(vae, cdl, clf_defended, Xatt, yatt, n_dims=10, n_iters=40, budget=2.5)
        round_history.append({"round": r + 1, "val_acc": round_val_acc, "attack_success_rate": round_asr,
                               "round_time_sec": round(time.time() - t_round, 1)})
        print(f"round {r+1}/{n_rounds}: val_acc={round_val_acc:.4f}  attack_success_rate={round_asr:.4f}  "
              f"({time.time()-t_round:.0f}s)")

    clean_acc_defended = evaluate(clf_defended, Xte, yte)
    asr_defended_final, _, _ = run_attack(vae, cdl, clf_defended, Xatt, yatt, n_dims=10, n_iters=40, budget=2.5)

    total_time = time.time() - t0
    results = {
        "dataset": "Fashion-MNIST (real, Zalando product photos, 10 apparel categories)",
        "defense_strategy": "fixed 4000-image adversarial pool, re-attacked every round with the "
                             "full-strength (eval-matching) CDL+SA attack, 6 rounds, 3 epochs/round",
        "total_time_sec": round(total_time, 1),
        "undefended_clean_test_acc": clean_acc_undefended,
        "undefended_attack_success_rate": asr_undefended,
        "defense_rounds": round_history,
        "defended_clean_test_acc": clean_acc_defended,
        "defended_attack_success_rate": asr_defended_final,
    }
    with open("results/robust_cnn_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(clf_defended.state_dict(), "results/clf_defended_v2.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
