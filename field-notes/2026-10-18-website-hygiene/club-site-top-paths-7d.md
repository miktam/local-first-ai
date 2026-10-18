# Second site (a sports club, not named) — top paths, 7 days to 2026-10-18

Source: Cloudflare dashboard → zone → Analytics → Requests, "Previous 7 days", top paths panel, captured 2026-10-18. The site moved from WordPress.com to Cloudflare on 2026-10-11, so this window is its first week as a static site.

| Path | Requests | What it is |
|---|---|---|
| /xmlrpc.php | 4.45k | WordPress remote-login endpoint; bots use it to guess passwords in bulk |
| / | 1.22k | homepage |
| /robots.txt | 451 | crawlers |
| /sitemap.xml | 292 | crawlers |
| /wp-login.php | 127 | WordPress login page |
| /this_is_a_new_hello_world.php | 120 | probe for a backdoor left by an earlier compromise |
| /opsh.php | 111 | same |
| /news-sitemap.xml | 108 | the old WordPress news sitemap; the site 301s it to /sitemap.xml |
| /wp-content/plugins/hellopress/wp_filemanager.php | 108 | probe for a known plugin backdoor |

Responses checked 2026-10-18: every `.php` path above returns 404 to GET and 405 to POST.

Mail records added the same day (the domain receives no mail): `v=spf1 -all`, `_dmarc` `v=DMARC1; p=reject;`, and a null MX (`0 .`).
