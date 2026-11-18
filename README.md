# robust-product-cnn

Attacks and (attempts to) defend a real e-commerce product-image classifier
(Fashion-MNIST: 70,000 real Zalando catalog photos, 10 apparel categories)
using two separate published methods, combined deliberately:

1. **Chivukula, Yang, Liu, Zhu, Zhou, "Game Theoretical Adversarial Deep
   Learning With Variational Adversaries," IEEE TKDE 2021** — a VAE as the
   adversary's generative strategy space, in a Stackelberg game against the
   classifier. The search alternates simulated annealing between the
   latent mean vector and the latent variance vector (their Algorithm 2,
   which they call "ALS" despite doing no least-squares fit), with a
   payoff of classifier error minus a cost penalty on the manipulation's
   size (read from the full paper text, not just the abstract).
2. **Zhu, Hu, Zhang, Zuo, "Coupled Dictionary Learning for Unsupervised
   Feature Selection," AAAI 2016** — a synthesis/analysis dictionary pair
   over the VAE's latent space, with L2,p-norm group-sparsity
   regularization on the analysis dictionary, used here to rank which
   latent dimensions matter and restrict the Stackelberg search to them.

These are genuinely two different papers solving two different problems;
combining them (CDL picks *where* to attack, Chivukula's method picks
*how*) is this project's own idea, stated as such throughout the code, not
presented as one paper doing both.

## Real use case

An attacker perturbs a product photo just enough to get it miscategorized
(wrong class, not one specific target — the realistic threat for a
catalog, as opposed to the original paper's one-specific-pair setup).

## What's honestly NOT faithful to the papers

- Zhu et al.'s exact optimizer is a proven-convergent IRLS algorithm. The
  AAAI proceedings PDF is behind a bot-blocker I couldn't get past, so I
  don't have their update rule — this uses ordinary gradient descent on
  the same objective instead.
- Chivukula et al. attack one specific class into one specific target
  class; this attacks toward whichever class is cheapest to reach.

## Results

See `BENCHMARKS.md` for the full real run (real Fashion-MNIST, real
training, ~53 minutes of CPU time, honestly reported including where the
defense did not work).

## Run it

```
pip install torch numpy
cd src && python train.py
```

The script checkpoints after every stage (VAE, CDL, undefended classifier,
and after every defense round) to `results/` and resumes automatically if
interrupted and rerun.
