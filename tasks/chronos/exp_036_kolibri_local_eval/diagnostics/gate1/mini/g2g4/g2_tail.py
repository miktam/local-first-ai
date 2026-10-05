"""Is one 6.08-sigma disagreement out of 8448 consistent with a bf16 noise model?
Model: the 6th-7th reference gap g has locally flat density near 0; the port
flips the pair when its error on (biased_7 - biased_6) exceeds g. That error is
N(0, (c*s_t*sigma_l)^2) with c = sqrt(2) for independent per-expert errors and
s_t a per-token scale (E[s^2] = 1 because sigma_l is an RMS over tokens).
Then, among disagreements, P(g > t sigma) = E_s[s * psi(t/(c s))] / E_s[s * psi(0)]
with psi(u) = phi(u) - u Q(u) (the integral of Q from u to inf)."""
import json, numpy as np
import math
class norm:
    @staticmethod
    def pdf(u): return np.exp(-0.5*np.asarray(u, dtype=float)**2)/math.sqrt(2*math.pi)
    @staticmethod
    def sf(u): return 0.5*np.vectorize(math.erfc)(np.asarray(u, dtype=float)/math.sqrt(2))
P3 = '$KIT/results/gate/20261005T050112Z/phase3.json'
nb = json.load(open(P3))['data']['g2']['natural_bf16']
n = np.array([r['n_disagree'] for r in nb]); mx = np.array([r['max_gap_over_sigma'] for r in nb])
N = n.sum()
def psi(u): return norm.pdf(u) - u * norm.sf(u)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80); gh_w = gh_w / gh_w.sum()
def tail(t, c, tau):
    # s lognormal, log s ~ N(mu, tau^2), E[s^2]=1 -> mu = -tau^2
    s = np.exp(-tau**2 + tau * gh_x)
    return np.sum(gh_w * s * psi(t / (c * s))) / np.sum(gh_w * s * psi(0.0))
def loglik(c, tau):
    # likelihood of the per-layer maxima: max of n_l iid draws with survival S(t)
    ll = 0.0
    for nl, m in zip(n, mx):
        h = 1e-3
        F1 = (1 - tail(m + h, c, tau)) ** nl; F0 = (1 - tail(m - h, c, tau)) ** nl
        ll += np.log(max((F1 - F0) / (2 * h), 1e-300))
    return ll
print(f'N disagreements {N}, layers {len(n)}; observed maxima >4: {(mx>4).sum()}, >5: {(mx>5).sum()}, >6: {(mx>6).sum()}')
for c, tau in [(np.sqrt(2), 0.0), (1.0, 0.0), (np.sqrt(2), 0.2), (np.sqrt(2), 0.3), (np.sqrt(2), 0.4), (np.sqrt(2), 0.5), (1.0, 0.4)]:
    e4 = sum(1 - (1 - tail(4, c, tau)) ** nl for nl in n)
    e5 = sum(1 - (1 - tail(5, c, tau)) ** nl for nl in n)
    p6 = 1 - (1 - tail(6, c, tau)) ** N
    print(f'c={c:.3f} tau={tau:.2f}: E[#layers max>4]={e4:5.1f} E[#>5]={e5:5.2f} '
          f'E[#dis>6sig]={N*tail(6,c,tau):6.3f} P(any>6sig)={p6:.3f} loglik={loglik(c,tau):8.2f}')
# MLE over a grid
best = max(((loglik(c, t), c, t) for c in np.linspace(0.8, 2.0, 25) for t in np.linspace(0, 0.8, 33)))
ll, c, t = best
print(f'MLE c={c:.3f} tau={t:.3f} loglik={ll:.2f}: E[#dis>6sig]={N*tail(6,c,t):.3f}, P(any>6sig)={1-(1-tail(6,c,t))**N:.3f}')
# the same with c fixed at sqrt(2)
best = max(((loglik(np.sqrt(2), t), t) for t in np.linspace(0, 0.8, 81)))
print(f'MLE at c=sqrt2: tau={best[1]:.3f} loglik={best[0]:.2f}: P(any>6sig)={1-(1-tail(6,np.sqrt(2),best[1]))**N:.3f}')
# what per-token scale is needed for the 6.08 event to be a 3-sigma_diff event
print('6.08 sigma_l = %.2f sigma_diff at c=sqrt2; = 3 sigma_diff if that token has s = %.2f' % (6.08/np.sqrt(2), 6.08/np.sqrt(2)/3))
print('--- wider grid')
grid = [(loglik(c, t), c, t) for c in np.linspace(0.2, 2.0, 37) for t in np.linspace(0, 1.4, 57)]
grid.sort(reverse=True)
for ll, c, t in grid[:5]:
    print(f'c={c:.3f} tau={t:.3f} loglik={ll:.2f} E[#>6sig]={N*tail(6,c,t):.3f} P(any>6)={1-(1-tail(6,c,t))**N:.3f} '
          f'E[#layers>4]={sum(1-(1-tail(4,c,t))**nl for nl in n):.1f} E[#layers>5]={sum(1-(1-tail(5,c,t))**nl for nl in n):.2f}')
# profile: for each tau, best c
print('--- profile over tau (best c each)')
for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    ll, c = max((loglik(c, t), c) for c in np.linspace(0.2, 2.0, 73))
    print(f'tau={t:.2f} c*={c:.3f} loglik={ll:8.2f} P(any>6sig)={1-(1-tail(6,c,t))**N:.3f} E[#layers>5]={sum(1-(1-tail(5,c,t))**nl for nl in n):.2f}')
# observed vs predicted quantiles of per-layer max at the tau=0 best c
ll, c0 = max((loglik(c, 0.0), c) for c in np.linspace(0.2, 2.0, 181))
print(f'homoscedastic best c={c0:.3f}; observed sorted maxima:', np.round(np.sort(mx), 2))
rng = np.random.default_rng(0)
