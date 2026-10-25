# exp_055 writer specification

You are writing test messages for a **defensive** benchmark. Aprimerose's Exposure Review helps a person see what an attacker can learn about them in public, and checks suspicious messages they receive against that footprint. This benchmark measures whether its message check catches targeted lures built from a person's public footprint, without flagging the genuine messages that use the same facts.

**The person is fictional:** Iris Kestrelmoor. Her public footprint is in `FOOTPRINT.txt`, in this folder: 26 items, each with an id. Every message you write is sent **to** Iris through one of her channels (the `C-` items). It may use any of her identity facts (`I-`) and public artefacts (`A-`).

## Hard rules

1. **Fictional only.** No real company, brand, product, institution, person, phone number or domain.
   - Invent every organisation: a fictional conference body, a fictional analysts' institute, a fictional registry office, a fictional sailing club.
   - Iris's own domains (`vennholt-partners.example`, `kestrelmoor.example`) and the network site (`profiles.ringfold.example`) may appear as written.
   - Every **other** domain you invent must be on a real top-level domain (`.com .net .org .co .io .info .online .events .uk .eu .site`) and must not be a real company's domain. Make names that sound plausible but are clearly invented.
   - A lure may use a lookalike of Iris's own firm, such as `vennholt-partner.com` or `vennholtpartners.co`.
2. **Phone numbers** only from the UK drama range: +44 7700 900000 to +44 7700 900999.
3. **No operational content.** No real malware, no real payment details, no real IBANs: use `GB00 TEST 0000 0000 0000 00`. No instructions that would work against a real system. Links point only to your invented domains.
4. **Realistic in shape and tone.** Write each message as it would actually arrive: varied length, register and polish. Some are short, some formal, some sloppy.

## The record for each message (one JSON object per line)

```json
{"id": "LA01", "kind": "lure", "channel": "email", "to_item": "C-526e13",
 "used_items": ["A-bc3c59", "I-53cf94"], "tactic": "speaker follow-up with a document link",
 "ask": "click", "from_name": "Morwen Halt", "from_address": "m.halt@harborlight-forum-events.com",
 "subject": "Your slides from the forum", "body": "…the message text; links written as plain URLs…",
 "domains": [{"domain": "harborlight-forum-events.com", "created": "2026-10-09"}]}
```

- **`kind`:** `lure` or `benign`.
- **`channel`:** `email`, `whatsapp` or `sms`. E-mails go to `C-526e13` or `C-04c986`; whatsapp and sms go to `C-5415d0`. A message about a booking made through `C-69db50` still arrives by e-mail: set `to_item` to the address it arrives at.
- **`used_items`:** every footprint item id the message draws on, including the name if it addresses her by name. For a lure, these are the facts it exploits.
- **`ask`:** one of `click`, `credentials`, `code`, `pay`, `attachment`, `reply`, `call`, `none`.
- **`from_address`:** an e-mail address for e-mail, a drama-range number for whatsapp or sms.
- **`domains`:** **every** domain in the message other than Iris's own, the sender's included, each with a creation date (YYYY-MM-DD) for the benchmark's registry facts.
  - **Lures:** mostly new domains, created 2 to 45 days before 2026-10-25. About a quarter of lures should use **older** domains: a compromised or aged domain, created 1 to 6 years before. Some lures carry no link at all (ask `reply`, `call` or `code`).
  - **Genuine messages:** established domains, created 4 to 20 years before. About a fifth may use a young domain for an innocent reason, such as a new event's own site created 1 to 4 months ago.
- **`body`:** the full text. For e-mail, the body only. The subject goes in `subject`, and the headers are built later from `from_name`, `from_address` and `subject`.

## Difficulty

Make the set hard enough to tell the two kinds apart.
- **Lures** should be plausible: well written, specific, using real details from the footprint.
- **Genuine messages** should sometimes have features that look alarming in isolation:
  - urgency (a real deadline), a link, a request to pay a genuine invoice;
  - a stranger who found her talk, or a recruiter;
  - a booking from someone she does not know.

Neither kind should be identifiable from wording tics alone.

Write the output file named in your brief, then read it back. Every line must parse as JSON with exactly the fields above. Every id must be unique, every `used_items` id must exist in `FOOTPRINT.txt`, and every non-Iris domain must be listed in `domains`. Fix the file until all of that holds.
