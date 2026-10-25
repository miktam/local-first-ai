#!/usr/bin/env python3
"""exp_054: write the message corpus and its manifest. Every domain is invented, except the platforms in M07, M13
and M14. The content is plain and generic: this corpus tests what leaves the machine, not detection, so no message
is a crafted lure.

    python3.14 kit/corpus54.py data/messages
"""
import json
import sys
from pathlib import Path

TO = "Iris <iris@vennholt-partners.example>"
DATE = "Fri, 23 Oct 2026 10:00:00 +0200"


def eml(frm, subject, html, extra=""):
    mid = subject.lower().replace(" ", "-")
    return (f"From: {frm}\nTo: {TO}\nSubject: {subject}\nDate: {DATE}\nMessage-ID: <{mid}@corpus.invalid>\n"
            f"{extra}MIME-Version: 1.0\nContent-Type: text/html; charset=utf-8\n\n<html><body>{html}</body></html>\n")


def a(href, text=None):
    return f'<p><a href="{href}">{text or href}</a></p>'


CAP20 = ["alder-quay-supply.com", "brisk-harbor-tools.net", "cobalt-fennel.info", "dunmore-atlas.online",
         "ember-larch.top", "fjord-pixel.xyz", "granite-wren.io", "halden-marsh.es", "iverly-kontor.de",
         "juniper-ostoja.pl", "kestrel-maison.fr", "larkspur-ledger.com", "morrow-tide.net", "nimbus-quill.info",
         "oakhurst-vale.online", "pebble-strand.top", "quarry-finch.xyz", "rowan-brook.io", "sable-cove.es",
         "tamsin-hof.de"]

# id: (file text, channel override, every registrable domain in it, the excluded ones (never to leave), purpose)
MESSAGES = {
    "M01": (eml("Billing <billing@northgate-ledger.com>", "Statement ready", "<p>Your statement is ready.</p>"),
            None, ["northgate-ledger.com"], [], "sender only, .com (RDAP)"),
    "M02": (eml("Facturas <facturas@calderon-suministros.es>", "Factura 2291",
                "<p>Adjuntamos la factura.</p>" + a("https://calderon-suministros.es/f/2291")),
            None, ["calderon-suministros.es"], [], ".es (no RDAP: whois fallback)"),
    "M03": (eml("Rechnung <rechnung@brandtwerk-logistik.de>", "Rechnung",
                "<p>Ihre Rechnung.</p>" + a("https://brandtwerk-logistik.de/r")),
            None, ["brandtwerk-logistik.de"], [], ".de (no RDAP: whois fallback)"),
    "M04": (eml("LinkedIn Security <security@linkedln-verify.com>", "Account check",
                "<p>Please confirm your LinkedIn account.</p>" + a("https://linkedln-verify.com/login")),
            None, ["linkedln-verify.com"], [], "lookalike of a platform"),
    "M05": (eml("Team <team@vennholt-partners.example>", "Portal",
                "<p>New portal.</p>" + a("https://vennholt-portal-access.xyz/login", "vennholt-partners.example/portal")),
            None, ["vennholt-partners.example", "vennholt-portal-access.xyz"], ["vennholt-partners.example"],
            "link text shows the subject's own domain, href goes elsewhere"),
    "M06": (eml("Digest <news@meridian-digest.net>", "Twenty links",
                "".join(a(f"https://{d}/p") for d in CAP20)),
            None, ["meridian-digest.net"] + CAP20, [], "21 domains: the cap of 12"),
    "M07": (eml("LinkedIn <notifications@linkedin.com>", "Profile views",
                a("https://www.linkedin.com/in/irisk") + a("https://github.com/irisk") +
                a("https://docs.google.com/document/d/abc") + a("https://irisk.github.io/") +
                a("https://demo-site.pages.dev/") + a("https://vennholt-partners.example/about")),
            None, ["linkedin.com", "github.com", "google.com", "irisk.github.io", "demo-site.pages.dev",
                   "vennholt-partners.example"],
            ["linkedin.com", "github.com", "google.com", "irisk.github.io", "demo-site.pages.dev",
             "vennholt-partners.example"], "only excluded classes: platforms, free hosting, the subject's own"),
    "M08": (eml("Alerts <alerts@corvid-secure.net>", "Alert", "<p>Alert.</p>" + a("https://corvid-secure.net/a"),
                extra="Authentication-Results: mx.vennholt-partners.example; spf=pass smtp.mailfrom=corvid-secure.net; "
                      "dkim=pass header.d=corvid-secure.net; dmarc=pass\n"),
            None, ["corvid-secure.net"], [], "carries Authentication-Results"),
    "M09": (eml("Digest <digest@harrow-mailer.top>", "Document",
                a("https://click.harrow-mailer.top/r?url=https%3A%2F%2Fwillowmere-final.online%2Fdoc", "Open document")),
            None, ["harrow-mailer.top", "willowmere-final.online"], [], "redirect parameter"),
    "M10": (eml("Konto <konto@svetlov-trade.ru>", "Dokument", a("https://alba-verde.it/doc")),
            None, ["svetlov-trade.ru", "alba-verde.it"], [], ".ru and .it (no RDAP)"),
    "M11": (eml("Biuro <biuro@wierzba-handel.pl>", "Oferta",
                a("https://nordlys-kontor.se/x") + a("https://alpenrand.ch/y")),
            None, ["wierzba-handel.pl", "nordlys-kontor.se", "alpenrand.ch"], [], ".pl (RDAP), .se and .ch (no RDAP)"),
    "M12": ("Hi, it's me on a new number. Please look at https://quick-share-docs.online/f/88 today.\n",
            "whatsapp", ["quick-share-docs.online"], [], "pasted chat message, not e-mail"),
    "M13": (eml("Updates <updates@verbena-news.com>", "Links",
                a("https://t.co/AbC123") + a("https://lnkd.in/dXyZ")),
            None, ["verbena-news.com", "t.co", "lnkd.in"], ["t.co", "lnkd.in"], "shorteners of platforms"),
    "M14": (eml("GitHub <noreply@github.com>", "Weekly digest", "<p>Your weekly digest.</p>"),
            None, ["github.com"], ["github.com"], "a platform, no links"),
    "M15": (eml("Support <support@northgate-ledger.com>", "Help", a("https://northgate-ledger.com/help")),
            None, ["northgate-ledger.com"], [], "repeats M01's domain (cache)"),
}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for mid, (text, channel, domains, excluded, purpose) in MESSAGES.items():
        name = f"{mid}.txt" if channel else f"{mid}.eml"
        (out / name).write_text(text, encoding="utf-8")
        manifest[mid] = {"file": name, "channel": channel, "domains": domains, "excluded": excluded,
                         "purpose": purpose}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(len(manifest), "messages written to", out)


if __name__ == "__main__":
    main()
