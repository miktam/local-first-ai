# Build log

One line a day of building local AI tools on one Mac mini: the CasaSol bot, the lead triage, and the decision models behind them. Each entry says what changed, what prompted it, how it was checked, and what I learned.

The experiments in [`tasks/chronos/`](../tasks/chronos/) answer questions that need a number. This log is everything between them: small changes, shipped daily, with the reason written down.

---

## Entry format

```markdown
### YYYY-MM-DD

- **<what changed>.** Input: <real | test | reading> — <what prompted it>. Checked: <how>.
- …

**Learned:** <one or two facts about a model, the hardware or the tools, with the source>.
**Inputs:** real N · test N · reading N.
```

- **Real:** a lead, a sign seen on the street, a question from an agency, or a message to the bot from a real user.
- **Test:** a failure found by one of our own tests (usually a Chronos experiment on synthetic data).
- **Reading:** an article, paper or post (the library behind this log).

The counts are the point. A change that starts from something real is worth more than one that starts from a test, and a log where real inputs stay at zero shows that too.

## Weekly roll-up

Every Sunday, one line under the week's entries:

```markdown
**Week of YYYY-MM-DD:** <n> changes · inputs real N / test N / reading N · <the one thing worth remembering>
```

## Model notes

What we know about each model, one dated line per fact, each with its source: [`models/`](./models/). Measured here, or published by someone we name.

## What never goes in

- Any client's or agency's data, or anything from a sheet sent under a data-processing agreement.
- Names of people or agencies we talk to, and their contact details.
- Credentials, hostnames, addresses, or security configuration beyond what a published experiment already shows.
- Numbers from an experiment that isn't published yet.

A real input is described by its type and what it showed, never by who sent it.
