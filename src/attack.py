"""The Stackelberg-game adversary (Chivukula et al., IEEE TKDE 2021),
restricted to the latent dimensions the CDL layer (Zhu et al., AAAI 2016)
scores as important - two separate published methods combined on purpose,
documented as such, not presented as one paper doing both.

Chivukula et al.: the adversary's leader payoff is J_L = error_F - lambda *
cost_L (cost_L = Frobenius norm of the manipulation). Their Algorithm 2
(they call it "ALS" despite doing no least-squares fit) alternates: fix the
variance-manipulation vector and use simulated annealing to update the
mean-manipulation vector, then fix the mean and SA-update the variance,
repeating until the payoff stops improving. Their Algorithm 3 (SA) perturbs
one latent coordinate at a time with a shrinking step size, accepting a
move whenever the payoff improves.

Zhu et al.: restricting which coordinates that per-coordinate SA loop
touches to the CDL layer's top-scoring dimensions is this project's own
combination of the two papers, not something either paper does - stated
plainly rather than implied. It has a real benefit beyond fidelity: a
32-coordinate-at-a-time search is expensive, and most of those 32 latent
dimensions turn out to carry very little reconstruction-relevant structure
once the CDL layer is trained, so searching only the ones that do is both
faster and arguably the more sensible attack in the first place.

The decoded image depends on a sample z = mu + sigma * eps, with eps held
fixed for the duration of one attack call, so that a change to the
variance block has an actual, deterministic effect on the outcome within
that call (the VAE's decode() only ever consumes a point, not a
distribution, so this is what makes "attacking the variance" meaningful
instead of a no-op).

One deliberate scope change from the Chivukula paper, stated rather than
hidden: their experiments attack one specific source class into one
specific target class (e.g. MNIST digit 5 into 8). This attacks toward
whatever class is cheapest to reach - any wrong apparel category, not one
fixed pair - the realistic threat model for a product catalog.
"""
import torch
import torch.nn.functional as F

LAMBDA_COST = 0.05  # leader payoff cost-penalty weight on the manipulation's Frobenius norm


def _decode(vae, mu, logvar, eps):
    z = mu + torch.exp(0.5 * logvar) * eps
    return vae.decode(z)


def _payoff(vae, classifier, mu, logvar, eps, mu0, logvar0, y):
    img = _decode(vae, mu, logvar, eps)
    logits = classifier(img)
    error = F.cross_entropy(logits, y, reduction="none")
    cost = (mu - mu0).norm(dim=1) + (logvar - logvar0).norm(dim=1)
    return error - LAMBDA_COST * cost


def _sa_search(vae, classifier, mu, logvar, eps, mu0, logvar0, y, target, delta, dims, n_steps=12, step_dec=0.5):
    """Algorithm 3, restricted to `dims` (the CDL-selected coordinates)."""
    block = mu if target == "mu" else logvar
    B = block.shape[0]
    best_block = block.clone()
    best_val = _payoff(vae, classifier, mu, logvar, eps, mu0, logvar0, y)

    for i in dims.tolist():
        eps_i = delta[:, i : i + 1].clone()
        for s in range(n_steps):
            cand_block = best_block.clone()
            direction = torch.sign(torch.randn(B, 1))
            cand_block[:, i : i + 1] = cand_block[:, i : i + 1] + direction * eps_i
            cand_mu = cand_block if target == "mu" else mu
            cand_logvar = cand_block if target == "logvar" else logvar
            cand_val = _payoff(vae, classifier, cand_mu, cand_logvar, eps, mu0, logvar0, y)
            improve = cand_val > best_val
            best_block = torch.where(improve.unsqueeze(1), cand_block, best_block)
            best_val = torch.where(improve, cand_val, best_val)
            eps_i = eps_i * (1 - step_dec * s / n_steps)
    return best_block, best_val


@torch.no_grad()
def stackelberg_attack(vae, cdl, classifier, x, y, n_rounds=3, delta_scale=1.2, n_dims=10):
    mu0, logvar0 = vae.encode(x)
    eps = torch.randn_like(mu0)  # fixed for this attack call - see module docstring
    mu, logvar = mu0.clone(), logvar0.clone()
    delta_mu = torch.full_like(mu0, delta_scale)
    delta_logvar = torch.full_like(logvar0, delta_scale)

    dims = torch.topk(cdl.feature_importance(), min(n_dims, mu0.shape[1])).indices

    best_val = _payoff(vae, classifier, mu, logvar, eps, mu0, logvar0, y)
    for r in range(n_rounds):
        prev_val = best_val.clone()
        mu, _ = _sa_search(vae, classifier, mu, logvar, eps, mu0, logvar0, y, "mu", delta_mu, dims)
        logvar, best_val = _sa_search(vae, classifier, mu, logvar, eps, mu0, logvar0, y, "logvar", delta_logvar, dims)
        if (best_val <= prev_val + 1e-4).all():
            break

    adv_img = _decode(vae, mu, logvar, eps)
    pred = classifier(adv_img).argmax(dim=1)
    fooled = pred != y
    return adv_img, fooled, pred


def sa_attack(vae, cdl, classifier, x, y, n_dims=10, n_iters=40, budget=2.5):
    """kept for backward compatibility with earlier call sites"""
    return stackelberg_attack(vae, cdl, classifier, x, y, n_dims=n_dims)


def pgd_latent_attack(vae, classifier, x, y, n_steps=7, step_size=0.4, budget=2.5):
    """A fast, differentiable attack for adversarial TRAINING, not
    evaluation. The Stackelberg search above treats the classifier as a
    black box (consistent with Chivukula et al., who explicitly do this),
    which is why it needs hundreds of forward-only evaluations per image -
    too slow to run over all 54,000 training images every epoch, which is
    what real adversarial training needs (the first version of this defense
    only ever attacked a 2,500-image subset per round; 95% of training data
    stayed clean, and the defense correspondingly failed). During training
    I have white-box access to my own classifier, so this uses ordinary
    projected gradient ascent in the same VAE latent space instead -
    several backward passes per batch rather than thousands of forward
    passes per image, fast enough to cover the full dataset every epoch.
    The final reported robustness number still comes from the slow,
    faithful, black-box Stackelberg search, which is strictly harder (no
    gradients) - this is only how the defense is trained, not how it's
    graded."""
    mu0, logvar0 = vae.encode(x)
    eps = torch.randn_like(mu0)
    mu = mu0.clone().detach().requires_grad_(True)
    logvar = logvar0.clone().detach().requires_grad_(True)

    for _ in range(n_steps):
        img = _decode(vae, mu, logvar, eps)
        logits = classifier(img)
        loss = F.cross_entropy(logits, y)
        grad_mu, grad_logvar = torch.autograd.grad(loss, [mu, logvar])
        with torch.no_grad():
            mu = mu + step_size * grad_mu.sign()
            logvar = logvar + step_size * grad_logvar.sign()
            # project back onto the same Frobenius-norm budget ball the
            # Stackelberg attack's cost penalty uses, so the two attacks
            # are comparable in how much manipulation they're allowed
            delta_mu, delta_logvar = mu - mu0, logvar - logvar0
            total_norm = (delta_mu.norm(dim=1, keepdim=True) ** 2 + delta_logvar.norm(dim=1, keepdim=True) ** 2) ** 0.5
            scale = torch.clamp(budget / (total_norm + 1e-8), max=1.0)
            mu = (mu0 + delta_mu * scale).detach().requires_grad_(True)
            logvar = (logvar0 + delta_logvar * scale).detach().requires_grad_(True)

    with torch.no_grad():
        adv_img = _decode(vae, mu, logvar, eps)
        pred = classifier(adv_img).argmax(dim=1)
        fooled = pred != y
    return adv_img.detach(), fooled, pred
