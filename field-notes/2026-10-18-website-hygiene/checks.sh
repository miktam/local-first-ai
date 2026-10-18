#!/usr/bin/env bash
# Read-only hygiene checks for a website behind Cloudflare.
# Usage: bash checks.sh example.com
# Makes DNS queries and plain HTTP GET/HEAD requests only. Run it against domains you own.
set -u
D="${1:?usage: bash checks.sh example.com}"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
code() { curl -s -o /dev/null -m 15 -w "%{http_code} %{content_type} %{redirect_url}" "$@"; }

echo "# checks.sh $D — $(date -u +%Y-%m-%dT%H:%MZ)"

echo; echo "## DNS"
NS=$(dig +short NS "$D" | head -1)
echo "nameserver:      $NS"
echo "MX:              $(dig +short MX "$D" @"$NS" | sort -n | tr '\n' ' ')"
echo "SPF:             $(dig +short TXT "$D" @"$NS" | grep -i 'v=spf1')"
echo "SPF records:     $(dig +short TXT "$D" @"$NS" | grep -ci 'v=spf1') (must be 1)"
echo "DMARC:           $(dig +short TXT "_dmarc.$D" @"$NS")"
echo "DKIM google:     $(dig +short TXT "google._domainkey.$D" @"$NS" | head -c 40)"
echo "DS at parent:    $(dig +short DS "$D" @1.1.1.1 | head -1)"
if dig +dnssec "$D" @1.1.1.1 | grep -q 'flags:.* ad'; then echo "DNSSEC:          validated (AD flag via 1.1.1.1)"; else echo "DNSSEC:          not validated"; fi

echo; echo "## Redirects"
echo "http://$D/          $(code "http://$D/")"
echo "https://www.$D/     $(code "https://www.$D/")"

echo; echo "## Paths (plain URL, then cache-busted)"
for p in / /no-such-page /wp-login.php /robots.txt /sitemap.xml /wrangler.toml; do
  printf "%-16s plain:  %s\n" "$p" "$(code "https://$D$p")"
  printf "%-16s busted: %s\n" "$p" "$(code "https://$D$p?v=$RANDOM$RANDOM")"
done

echo; echo "## Security headers on /"
curl -sI -m 15 "https://$D/" | tr -d '\r' | grep -i -E '^(strict-transport-security|x-frame-options|content-security-policy):' || echo "(none)"

echo; echo "## External hosts referenced by the homepage (browser-like request)"
curl -s -m 15 -A "$UA" -H "Accept: text/html" "https://$D/" | grep -o -E 'https?://[a-zA-Z0-9.-]+' | sort | uniq -c | sort -rn

echo; echo "## Cloudflare Web Analytics beacon"
echo "plain curl:          $(curl -s -m 15 "https://$D/" | grep -c cloudflareinsights) occurrence(s)"
echo "browser-like request: $(curl -s -m 15 -A "$UA" -H "Accept: text/html" "https://$D/" | grep -c cloudflareinsights) occurrence(s)"
