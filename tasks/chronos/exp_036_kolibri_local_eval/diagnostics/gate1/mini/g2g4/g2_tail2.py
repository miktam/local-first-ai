exec(open('$SCRATCH/gatediag_g2g4/g2_tail.py').read().split("print(f'N disagreements")[0])
keep = np.arange(len(n)) >= 2
n, mx = n[keep], mx[keep]; N = n.sum()
print(f'layers 2-49: N={N}, maxima>4: {(mx>4).sum()}, >5: {(mx>5).sum()}, >6: {(mx>6).sum()}')
for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    ll, c = max((loglik(c, t), c) for c in np.linspace(0.2, 2.0, 73))
    print(f'tau={t:.2f} c*={c:.3f} loglik={ll:8.2f} E[#>6sig]={N*tail(6,c,t):.3f} P(any>6sig)={1-(1-tail(6,c,t))**N:.3f} '
          f'E[#layers>4]={sum(1-(1-tail(4,c,t))**nl for nl in n):.1f} E[#layers>5]={sum(1-(1-tail(5,c,t))**nl for nl in n):.2f}')
