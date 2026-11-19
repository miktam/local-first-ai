# Thirteen runs of an agent loop: who caught the defects

*2026-11-19. From 6 to 10 November 2026 a cloud-model session on this project ran a loop: it wrote a contract and a check for a piece of work, froze the check in git, handed the work to a second agent, and gave the result to a third that never saw the second's report. Thirteen runs closed. This note publishes the counts and the table behind them. The models are Anthropic's, not local ones. Seven of the thirteen runs built the kits, graded the answers or wrote the results of [exp_056](../../tasks/chronos/exp_056_triage_injection/) and [exp_061](../../tasks/chronos/exp_061_value_injection/), so this is also a record of how those results were checked.*

---

## The set-up

Four roles, three of them models.

- **The dispatcher** picks the next item, writes the brief (the goal, the contract, the paths that may change) and an acceptance check. The check is committed before any work starts and must fail on the unchanged state.
- **The worker** does the work against that check. It cannot edit the check.
- **The reviewer** sees the result and the contract, never the worker's report, and answers yes or no with objections. Objections that hold go back to the worker once.
- **Andrei** decides every outward step: push, deploy, send, publish. Each waits for his word. Once, in run 12, he gave it in advance and the dispatcher pushed.

Each run ends in one row of a table and at most two written lessons. The first run is in the build log for [6 November](../../log/2026-11.md).

## The measure, fixed before any result

Three counts, written down the evening the first run was registered, and the bar: the loop is kept if the first is at least 4 and the third at most 1.

| Count | Result |
|---|---|
| 1. Runs the loop started and finished with no new prompt, where Andrei's only part was a yes or no at the gate | **5** of 13. The first of the five was started on his "go", which the rule counted in advance. The other 8 ran inside work he had opened: runs 04 to 07 in a stream he opened with one instruction, runs 10 to 13 after his instruction, decision or delegation |
| 2. Runs where the check or the reviewer found a defect the worker's report did not mention | **11** of 13 by that wording. In one of the other two the catch was the worker's own stop; in the other the review found points of framing and no defect. Counted loosely, as "something was caught", it is 13 of 13 |
| 3. Runs Andrei had to stop, redo or correct | **0** |

The bar uses counts 1 and 3 and is met: 5 and 0. Andrei kept the loop on 11 November, with the four changes below.

## The thirteen runs

| Run | Date | Kind | Reviewer's first answer | Blocking | Should-fix | Minutes |
|---|---|---|---|---|---|---|
| 01 | 6 Nov | code | no | 1 | 7 | 566 |
| 02 | 7 Nov | code | yes | 0 | 5 | 59 |
| 03 | 8 Nov | code | no | 1 | 7 | 42 |
| 04 | 8 Nov | experiment kit | no | 1 | 9 | 72 |
| 05 | 8 Nov | experiment kit | yes | 0 | 0 | 81 |
| 06 | 8 Nov | grading | yes on conditions | 0 | 4 | 37 |
| 07 | 8 Nov | text | yes on conditions | 0 | 5 | 20 |
| 08 | 9 Nov | code | no | 2 | 2 | 41 |
| 09 | 10 Nov | code | yes | 0 | 3 | 69 |
| 10 | 10 Nov | experiment kit | no | 3 | 5 | 79 |
| 11 | 10 Nov | text | no | 1 | 6 | 114 |
| 12 | 10 Nov | code | no | 2 | 4 | 89 |
| 13 | 10 Nov | text | no | 3 | 9 | 93 |

The reviewer's first answer was no in 8 runs. Eleven runs had a fix round, and two of those fix rounds introduced a new defect (runs 03 and 08). Minutes run from the commit that froze the check to the commit that closed the run; run 01 sat overnight. The median is 72. What each run was is in [`runs.tsv`](./runs.tsv).

"Blocking" and "should-fix" are the reviewer's own labels. That every blocking finding came from the reviewer is true by construction and is not a result.

## Who caught what

[`defects.tsv`](./defects.tsv) has one row per blocking finding, per defect a run's verdict names, per error in a check, a brief or a registration, and per catch by the worker or the dispatcher. Findings of one kind from one run, caught by the same role, share a row. Other should-fix items are not listed. That gives 51 rows.

| Caught by | Rows |
|---|---|
| The reviewer alone | 29 |
| The worker and the reviewer | 9 |
| The worker alone | 8 |
| The dispatcher's own extra check | 4 |
| The frozen check | 1 |
| Andrei | 0 |

| Kind | Rows | What it is |
|---|---|---|
| Code | 18 | a defect in the worker's code |
| Contract | 11 | the dispatcher's error in a brief, a design or a registration |
| Fact | 9 | a wrong fact or number in a text |
| Check | 8 | a defect in the frozen check |
| Process | 3 | process or privacy |
| Other | 2 | the reviewer's own error; Andrei's first build of a test set |

Three readings.

- **19 of the 51 rows are the dispatcher's own errors**, contract and check together. 18 rows are defects in the worker's code. A row can group several findings, and the 9 fact rows sit in texts the workers wrote, so the rows do not give a total per role. They do show that the role writing the task and its test was itself a large source of errors.
- **One row of 51 is credited to the frozen check**, and the worker was the one who ran it. The check's other job came first: it fixed what "done" meant before the work started, and it had to fail on the unchanged state before the worker was given the task.
- **No check case read a summary sentence.** Cases held numbers to the record; none tested what a sentence claimed. All 9 fact rows are of that kind. The reviewer caught 7 of them, in reviews that traced or recomputed every number. Reading each summary line against the numbers beside it became a written rule after run 13.

Where a missed defect would have gone: 20 rows were headed for this public repository. 4 of those reached it anyway, and a fifth is a commit subject. Three of the four were caught before the push and went out as they stood, in notes pinned by hash; one was caught after it. Six were headed for a document handed back to the person who sent one in.

## The frozen checks were wrong in 7 of 13 runs

| Run | Check score | What was wrong |
|---|---|---|
| 03 | 6/6 | Passed while the rule it guarded was broken: its "same mail" case re-added identical bytes, so the rule on message ids never ran |
| 08 | 8/8 | Passed while false: it asserted the new label and never searched for the words |
| 09 | 17/17 | Passed while false: it held only the reported shape, and missed bodies that end in a line break |
| 10 | 21/21 | Passed while false: it held the brief's own command, which left an option out |
| 12 | 16/16 | Passed while false: its stand-in for the model answered alike on every view of a message |
| 04 | 5/6 | Failed on a correct result: one case's fixtures held the words it banned |
| 13 | 15/16 | Failed on a correct result: a row prefix came from a shortened quotation |

In runs 08, 10 and 12 the check shared the blind spot of the contract it was written from. Both came from the dispatcher. The lessons about checks began with run 01, and a wrong check still appeared in run 13.

## What changed on 11 November

1. The reviewer reads the contract and the frozen check before the worker starts.
2. Every check case is shown failing on a bad form of the thing it guards, and passing on a correct one, before it is frozen.
3. Every time in the record is read from the clock or from a commit.
4. Each run records which model filled each role.

Two smaller rules were added the same afternoon: how a brief words what the worker may change, and a list in every review of changes nobody asked for.

The first run under the changed rules closed on 13 November. It is one case and not a rate. The dispatcher proved its check against one broken form per case, each of its own choosing, and the proof passed 14 of 14. The reviewer then took twelve other rules out of the same reference, and the check still passed 14 of 14. The check was rewritten to 28 cases, 26 of them proved against 35 broken forms, before any work started. The worker then passed it whole on the first round, and the review of the result had no blocking finding. That reviewer did not read the fix round. A proof built from the author's own wrong forms measured the author's imagination.

## A later run: a fix that passed its tests by stepping aside

On 17 and 18 November the loop was pointed at a public chat bot that answers property searches from a set of listings. Twelve reviewing agents, one per angle, read and ran the code on a copy. Each finding rated high or medium went to a second agent told to refute it, and 90 stood, each with a test that failed on the unchanged code. Those 90 tests were written by the agents that found the faults, and were frozen. Two workers fixed the code against 63 of them; the other 27 were set aside for a later run or for the owner's decision. The numbers are in [`night_run.tsv`](./night_run.tsv).

One part of the bot answers a search in code and steps aside when it cannot read the question, which then goes to the language model without the list of homes. After the fix, the frozen tests for that part stood at 253 of 254. The reviewer was shown the diff and not the workers' reports. It ran the old code and the new side by side on 72 everyday wordings of a search, each with one extra detail, and on 49 of them the new code no longer gave the list of homes the old code gave. On 107 plain searches with a single filter, 4 lost the list and 3 gained one. The reviewer's estimate was that a quarter to a third of real searches would get no list. Many of the frozen tests had been written to return early when that part of the bot gave no answer, so a fix that stepped aside on any doubt passed them. After the one fix round the dispatcher re-ran the reviewer's comparison: 5 of 72, and each of the five either matches nothing or names a place no listing holds. The frozen tests read 253 of 254 before and after.

Limits of this run: the reviewer was Opus, not Fable as the rules name, because Fable had reached its usage limit. The reviewer did not read the fix round, and three of its should-fix points rest on the workers' own tests. No review of the frozen tests before the work is on record: they went to the workers a minute after they were frozen. Two workers ran at once, where the rules say one. The other frozen set the workers fixed against stood at 330 of 451 after the fix; some of the rest fail because the fix replaced the code they were anchored on, and the others cover work left for later runs, not listed here.

## What this record cannot show

- **An error all three roles share.** Dispatcher, worker and reviewer are models from one vendor, and the dispatcher writes both the contract and its check. Every row above was seen by at least one role. An error all three hold leaves no row, and nothing here measures that class. No second critic from another vendor was added.
- **The record is the dispatcher's.** The table of runs, the verdicts and the counts were written by the role being measured. No entry is Andrei's. When the record was compiled for his decision it contradicted itself in five places, among them times written inside the files of runs 01 to 05 that are later than the commits holding them. This note uses git times.
- **Which model did what.** Briefs gave only a model name, "Opus" or "Fable", and the dispatcher's model was not recorded. In run 06 the three graders and the reviewer were all Fable.
- **Cost.** There is no token count and no price per run. The 1,362 minutes are the sum of the rounded minutes above (the private record's own total reads 1,363). There is no separate figure for the dispatcher's own time or for Andrei's time at the gates.
- **A comparison.** No item was done without the loop, so nothing here says the loop produced better work than one session would have. It says what was found in work done under the loop.
- **Two saved reviews are shortened** (runs 11 and 13). Run 13's count of 9 comes from its verdict. Run 11's 6 is the shortened review's count; its fix round closed ten.

## What is withheld

The briefs, checks, reviews and verdicts stay in a private repository: they quote private code and working notes. So do the 90 findings of the later run and their tests. Three runs (03, 08, 09) built a package in a repository with no remote, and runs 01 and 02 changed code outside this repository. Their rows are described, not linked. Run 12 changed the message check, also private; exp_056's Amendment 4 pins its commit. Run 10's kit was built and corrected in a private repository and entered this one in run 11. The rows from runs 04 to 07, 11 and 13 are about files in this repository; the corrections they led to can be read in the history of the two experiment folders, the Chronos log and the front page.

## Reproduce the counts

```
python3 count.py
```

Standard library only. It reads the trial's three `.tsv` files and prints every count that comes from them. `night_run.tsv` holds the later run's figures as they are quoted above.
