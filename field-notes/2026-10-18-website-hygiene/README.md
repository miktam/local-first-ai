# Field note: aprimerose.com website hygiene (2026-10-17/18)

**What this is:** the record behind the post *80% of Our Website's Traffic Was a Dead Subdomain Redirecting to Itself* ([localfirstai.eu](https://localfirstai.eu/posts/2026-10-18-dead-subdomain-redirecting-to-itself/)).

**What this is not:** a Chronos experiment. This was an operational clean-up, recorded after the fact. No hypothesis was registered beforehand, and the figures are observations from production traffic, not measurements under controlled conditions. Claims are limited to what was observed.

Recorded by Nestor (an AI coding assistant session, Claude) for Andrei.

## Files

| File | Contents |
|---|---|
| [`analytics-before-7d.md`](analytics-before-7d.md) | Cloudflare zone analytics, 7 days to 2026-10-17 (before the fixes) |
| [`analytics-after-6h.md`](analytics-after-6h.md) | The same view, 6 hours on 2026-10-18 (after the fixes) |
| [`club-site-top-paths-7d.md`](club-site-top-paths-7d.md) | Top paths for the second site in the post (a sports club, not named) |
| [`checks.sh`](checks.sh) | The read-only checks. Run `bash checks.sh aprimerose.com`, or point it at a domain you own |
| [`results-2026-10-18.txt`](results-2026-10-18.txt) | `checks.sh` output against aprimerose.com after the fixes |

## Claims in the post and their evidence

| Claim | Evidence |
|---|---|
| 230,680 requests in 7 days; 200,170 301s; 186,270 with no user agent; 185,560 for `app.aprimerose.com` | `analytics-before-7d.md` |
| The callers were machines in Microsoft's cloud | `analytics-before-7d.md`: 9 of the top 9 client IPs `whois` to Microsoft; the top countries match Azure regions |
| 84% fewer requests, 98% fewer redirects | `analytics-after-6h.md`, last section (arithmetic shown) |
| `app.aprimerose.com` redirected to itself | Observed 2026-10-17, before the records were deleted (not re-runnable now): `curl -D - https://app.aprimerose.com/` returned `301` with `location: https://app.aprimerose.com/`; `curl -L --max-redirs 30` stopped with "Maximum (30) redirects followed". The origin (Firebase Hosting) presented a `firebaseapp.com` certificate and returned "Site Not Found" over HTTPS, and `301` to HTTPS over plain HTTP |
| 25 DNS names deleted | Categories below; names withheld |
| SSL/TLS "Automatic" was running "Flexible"; now Full (strict) | Dashboard, 2026-10-18: "Automatic SSL/TLS (recommended) — Currently running: Flexible", then set to Full (Strict) |
| Google Analytics and Google Fonts on every page; removed | Public site repo [aprimerose/website](https://github.com/aprimerose/website), commit [`a3540dd`](https://github.com/aprimerose/website/commit/a3540ddda4e9046534a386cf9aa4949e91b73bea) (the diff shows both on every page); `results-2026-10-18.txt`, "External hosts" |
| No 404 page: unknown paths, `/robots.txt` and `/sitemap.xml` returned the homepage with 200 | Observed 2026-10-17: `/1.php`, `/wp-login.php`, `/robots.txt`, `/sitemap.xml` each returned `200`, 3,168 bytes, `text/html`. Fix: commit [`db045c8`](https://github.com/aprimerose/website/commit/db045c83e3f0e5bd027fd2663516eae250720385); now `results-2026-10-18.txt`, "Paths" |
| The deploy published a build configuration file | Observed 2026-10-17: `/wrangler.toml` returned `200`, 77 bytes, `application/toml`. Removed in [`dbde9d2`](https://github.com/aprimerose/website/commit/dbde9d2b439c99a669de93d1b97d8f7d7acad8ab) |
| `/.dev.vars` was the most-requested missing path, 40 in 6 hours | `analytics-after-6h.md`, "Top paths" |
| A ten-hour-old cached copy was served on the plain address | Observed 2026-10-18: plain `/wrangler.toml` → `200`, `age: 37149`, `cache-control: public, s-maxage=604800`; with `?v=` → `404`. Still visible in `results-2026-10-18.txt` |
| Web Analytics had been running; plain `curl` didn't show the beacon | `results-2026-10-18.txt`, last section (0 vs 1 occurrence); dashboard shows the site created four years earlier, automatic setup |
| Email before: SPF `?all`, DMARC `p=none` with no reports, no Google DKIM | Observed 2026-10-17 with `dig`: `v=spf1  include:spf.mandrillapp.com  include:_spf.google.com  ?all`; `v=DMARC1; p=none`; no `google._domainkey` record |
| Email after: DKIM 2048-bit, SPF `~all`, DMARC reporting; a test message passes all three | `results-2026-10-18.txt`, "DNS". Test message 2026-10-18: `spf=pass`, `dkim=pass (2048-bit key) header.s=google`, `dmarc=pass (p=none)` (headers kept privately) |
| DNSSEC was off; on now | Before: no DS record at the `.com` registry (2026-10-18). After: `results-2026-10-18.txt`, "DNSSEC: validated" |
| Second site: `/xmlrpc.php` 4,450 requests vs 1,220 for the homepage | `club-site-top-paths-7d.md` |

## The 25 deleted DNS names, by category

| Category | Names |
|---|---|
| The dead Firebase app (`app`) | 1 (two A records) |
| Hostnames for old products and their CI | 6 |
| An old WordPress.com links page | 1 |
| CNAME to a deleted AWS load balancer, plus its certificate-validation record | 2 |
| Help and FAQ hostnames on services no longer used | 2 |
| A stale ACME challenge record | 1 |
| Skype for Business (service shut down by Microsoft in 2021) | 4 |
| Microsoft device-enrollment records | 2 |
| SendGrid (targets no longer resolve) | 3 |
| Records with the company name misspelled | 3 |
| **Total** | **25** |

## Withheld

Cloudflare account identifiers; client IP addresses (their owners are given instead); the names of deleted hosts other than `app`; full email headers; anything about infrastructure still in use.
