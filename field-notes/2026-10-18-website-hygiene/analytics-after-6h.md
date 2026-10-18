# aprimerose.com — Cloudflare zone analytics, 6 hours on 2026-10-18 (Sunday morning)

Source: Cloudflare dashboard → aprimerose.com → Analytics → Requests, "Previous 6 hours", copied 2026-10-18. Same view as `analytics-before-7d.md`. The fixes went live during the evening of 2026-10-17.

Edits: account identifiers removed; client IP addresses replaced by the organisation their `whois` record names. The "change" column compares with the previous 6 hours, not with the week before.

## Totals

| Metric | Value | Change shown by Cloudflare |
|---|---|---|
| Total requests | 1.3k | ↗ 47.7% |
| Total data transfer | 1.97 MB | ↘ 55.8% |
| Total visits | 29 | ↘ 86.3% |
| Cache hit rate | 0.84% | ↘ 42.7% |
| 5xx error rate | 0.00% | — |

## Requests by status code

| Class | Requests |
|---|---|
| 2xx | 55 |
| 3xx | 127 |
| 4xx | 1.12k |
| 5xx | 0 |

## Requests by cache status

Dynamic 1.1k · None 153 · Bypass 32 · Revalidated 11 · Miss 4

## Top hosts

aprimerose.com 1.29k · www.aprimerose.com 11

## Top paths

| Path | Requests |
|---|---|
| / | 57 |
| /.dev.vars | 40 |
| /robots.txt | 15 |
| /about/ | 11 |
| /app | 8 |
| /services/.env | 6 |
| /panel/.env | 6 |
| /secrets.env | 6 |
| /wp-login.php | 6 |

## Top client IPs (owner per `whois`)

| Owner | Requests |
|---|---|
| Google Cloud (`GOOGL-2`) | 434 |
| Amazon (`AMAZON`) | 316 |
| Microsoft (`MSFT`) | 171 |
| Microsoft (`MSFT`) | 101 |
| Microsoft (`MSFT`) | 49 |
| Microsoft (`MSFT`) | 46 |
| small hosting provider | 25 |
| M247 (hosting) | 21 |
| small hosting provider | 14 |

## Top countries

Netherlands 437 · United States 379 · Korea, South 220 · Australia 101 · India 54 · Sweden 27 · France 24 · Lithuania 22 · Singapore 6

## Top user agents

| User agent | Requests |
|---|---|
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/131.0.0.0 Safari/537.36 | 316 |
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/120.0.0.0 Safari/537.36 | 197 |
| None | 172 |
| Claude-User/1.0 | 58 |
| Mozilla/5.0 (Linux; Android 10; K) … EdgA/153.0.0.0 | 46 |
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 | 25 |
| Amazonbot/0.1 | 24 |
| YouBot/1.0 | 23 |
| ClaudeBot/1.0 | 22 |

Top browsers: Unknown/Others 635 · Chrome 587 · GoogleBot 26 · BingBot 18 · Edge 10 · Firefox 10 · AppleBot 6 · MobileSafari 5 · Safari 4

## Comparison used in the post

The 7-day view averages 230.68k / 28 = 8.24k requests and 200.27k / 28 = 7.15k redirects (3xx) per 6 hours. This 6-hour window had 1.3k requests (−84%) and 127 redirects (−98%). A Sunday-morning window against a weekly average is a rough comparison.
