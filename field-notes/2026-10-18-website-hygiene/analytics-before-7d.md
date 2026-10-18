# aprimerose.com — Cloudflare zone analytics, 7 days to 2026-10-17

Source: Cloudflare dashboard → aprimerose.com → Analytics → Requests, "Previous 7 days", copied 2026-10-17. This is the zone's HTTP traffic view, which counts every request (bots included). It is not Cloudflare Web Analytics.

Edits: account identifiers removed; client IP addresses replaced by the organisation their `whois` record names; hostnames of deleted DNS records withheld, except `app.aprimerose.com`. Figures are as the dashboard showed them (rounded to three significant digits by Cloudflare).

## Totals

| Metric | Value | Change shown by Cloudflare |
|---|---|---|
| Total requests | 230.68k | ↘ 6.5% |
| Total data transfer | 273.27 MB | ↘ 2.4% |
| Total visits | 22.23k | ↘ 8.5% |
| Cache hit rate | 0.41% | ↗ 188.5% |
| 5xx error rate | 0.01% | ↘ 97.9% |

## Requests by status code

| Class | Requests |
|---|---|
| 2xx | 25.38k |
| 3xx | 200.27k |
| 4xx | 5.02k |
| 5xx | 15 |

Top status codes: 301 Moved Permanently 200.17k · 200 OK 25.25k · 404 Not Found 3.19k · 403 Forbidden 1.22k · 405 Method Not Allowed 510 · 204 No Content 117 · 499 Client Closed Request 105 · 308 Permanent Redirect 91 · 206 Partial Content 13

## Requests by cache status

Dynamic 208.74k · None 19.62k · Miss 888 · Hit 871 · Expired 479

## Top hosts

| Host | Requests |
|---|---|
| app.aprimerose.com | 185.56k |
| aprimerose.com | 35.52k |
| www.aprimerose.com | 4.54k |
| [deleted legacy host A] | 4.01k |
| [deleted legacy host B] | 899 |
| aprimerose.com:8080 | 31 |
| aprimerose.com:2083 | 25 |
| aprimerose.com:8443 | 24 |
| [deleted legacy host C] | 21 |

## Top paths

| Path | Requests |
|---|---|
| / | 2.23k |
| /wp-content/plugins/hellopress/wp_filemanager.php | 2.11k |
| /this_is_a_new_hello_world.php | 2.06k |
| /1.php | 1.89k |
| /BDKR28WP.php | 1.54k |
| /class-walker-footer-dev.php | 1.54k |
| /opsh.php | 1.45k |
| /chosen.php | 1.38k |
| /3PJcpMFsD8B.php | 1.29k |

## Top client IPs (owner per `whois`)

All nine of the top nine client IPs belong to Microsoft (`OrgName: Microsoft Corporation` / `MSFT`). Their request counts: 21.47k, 15.26k, 15.12k, 12.19k, 11.88k, 11.64k, 9.01k, 8.99k, 8.73k.

## Top countries

India 77.22k · Australia 41.78k · Korea, South 35.62k · Hong Kong 16.91k · Japan 15.97k · United States 12.54k · Netherlands 8.88k · Germany 3.32k · Singapore 3.31k

## Top data centers

Mumbai (BOM) 77.19k · Sydney (SYD) 41.8k · Seoul (ICN) 35.56k · Hong Kong (HKG) 17.03k · Osaka (KIX) 15.53k · Amsterdam (AMS) 7.76k · Frankfurt (FRA) 4.71k · Atlanta (ATL) 4.7k · Paris (CDG) 4.68k

## Top user agents

| User agent | Requests |
|---|---|
| None | 186.27k |
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/123.0.0.0 Safari/537.36 | 5.28k |
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/120.0.0.0 Safari/537.36 | 4.05k |
| Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/131.0.0.0 Safari/537.36 | 3.34k |
| Go-http-client/1.1 | 2.2k |
| Go-http-client/2.0 | 1.73k |
| curl/8.7.1 | 1.5k |
| bingbot/2.0 | 972 |
| Mozilla/5.0 (Linux; Android 10; K) … EdgA/153.0.0.0 | 762 |

Top browsers: Unknown/Others 204.68k · Chrome 18.72k · Curl 1.5k · Firefox 1.49k · BingBot 997 · GoogleBot 925 · Opera 831 · AppleBot 530 · MobileSafari 305

Top operating systems: Unknown/Others 207.65k · Windows 18.53k · MacOSX 2.61k · Linux 824 · Android 698 · iOS 351 · ChromeOS 12

Top device types: Desktop 228.41k · Mobile 1.95k · Tablet 316
