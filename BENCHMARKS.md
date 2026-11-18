# Benchmarks (real run, real Fashion-MNIST, ~53 minutes CPU time)

70,000 real Zalando catalog photos (10 apparel categories): 54,000 train,
6,000 validation, 10,000 test. 600 held-out test images used for the
(expensive) adversarial search, never seen during training.

## Undefended classifier

- **Clean test accuracy: 91.94%** (15 epochs, converged — a solid,
  unremarkable number for a small CNN on this dataset, in line with
  published Fashion-MNIST baselines)
- **Attack success rate: 84.0%** — the Chivukula-style Stackelberg
  adversary, searching only the CDL-selected latent dimensions, fools this
  classifier on the large majority of attempted images

## Defense: 6 rounds of Stackelberg retraining (fixed 2,500-image pool,
re-attacked against the current classifier every round)

| round | clean val accuracy | attack success rate |
| --- | --- | --- |
| 1 | 92.05% | 100.0% |
| 2 | 91.72% | 100.0% |
| 3 | 91.85% | 97.5% |
| 4 | 92.30% | 100.0% |
| 5 | 92.08% | 98.7% |
| 6 | 92.18% | 100.0% |

- **Defended clean test accuracy: 91.83%** — essentially unchanged from
  undefended (91.94%). The defense doesn't hurt normal performance.
- **Defended attack success rate: 99.83%** (on a fresh, independent
  attack run against the final defended model) — essentially unchanged
  from the undefended 84.0%, and if anything slightly worse.

## Honest conclusion

**The defense did not work.** Six rounds of adversarial retraining,
against a fixed pool re-attacked every round (the fix for an earlier
version's non-convergent defense, where fresh random subsets each round
caused the attack rate to oscillate rather than settle), still left the
classifier essentially as vulnerable as it started. Clean accuracy never
suffered, so this isn't a case of the defense trading accuracy for
robustness — it's a case of the defense not conferring robustness against
this adversary at all, in this configuration.

The honest, specific hypothesis for why: the Stackelberg search here is
much stronger than a single-step gradient attack (FGSM) or even a
bounded-budget PGD attack — it's a from-scratch search across both the
latent mean and variance, restarted fresh against the current model every
round, with no budget cap tying it to what the previous round's defense
specifically patched. A classifier retrained on round N's adversarial
examples has no particular reason to be robust to round N+1's search,
which explores different directions each time. Three plausible next
levers, untried here: (1) a much larger and/or growing adversarial pool
across rounds instead of a fixed 2,500, (2) more epochs per round so the
retraining actually converges on the adversarial examples rather than
partially fitting them (loss was still falling at epoch 3 every round —
see `results/checkpoint.json`), (3) constraining the attack's cost budget
more tightly, since an unconstrained search will likely always find
something.

## How to reproduce

```
cd src && python train.py
```

Checkpoints after every stage in `results/` (resumes automatically).
Full per-round history: `results/robust_cnn_results.json`.
