# Author contact, before anything is published

**Status: templates only. Nothing has been sent, and no benchmark has been coded.**

This is step 6 of `SURVEY_PROTOCOL.md`, written out. It is a correctness step before
it is a courtesy one: the authors of a benchmark are the people most likely to know
about a power analysis, a rejection log, or a decontamination audit that exists and
was never published. Every such reply removes a row from the table. A table that did
not go through this is measuring what people wrote down, and quietly reporting it as
what they did.

The tone rule for all of it: **the finding is about a criterion, not about a person,
and RHOB failed the same diagnostics.** That second clause is not throat-clearing —
it is the reason this can be sent at all. `audit/SELF_CODING.md` shows RHOB
pre-audit with both of its equivalence-shaped criteria inverted.

---

## Sequence

| When | What |
|---|---|
| Day 0 | Send each author group **their rows only**. Never the comparative table. |
| Day 0–28 | Corrections arrive. Apply them to the sheet as they land. |
| Day 28 | Window closes. Publish, with responses next to the rows. |
| After | Corrections keep being applied. The sheet is versioned, not frozen. |

Four weeks is the default and is stated in the first email. If someone asks for
longer, give it — there is no deadline here that matters more than being right.

---

## What each group receives

One email, their rows only, and:

- the exact criterion text as coded, with the source location
- which diagnostic fired and why
- the remedy, concretely
- a link to the protocol and to RHOB's own coding
- the two questions

Never send a group another group's rows, and never send the comparative ranking. A
benchmark's authors should be able to check their own coding without being handed a
league table they did not ask to be in.

---

## Template: a criterion coded INVERTED

> Subject: A question about the `<criterion>` check in `<benchmark>`
>
> Dear <names>,
>
> I'm a student working on how benchmark preconditions are verified. I've been
> coding published benchmarks for one specific property — whether a criterion that
> asserts two things are *the same* is supported by a test that could have said they
> aren't — and `<benchmark>` came up. Before I publish anything I'd like to check
> whether I've read your work correctly, because I may well not have.
>
> The criterion I mean is `<criterion>`, at `<source>`:
>
> > <verbatim criterion text>
>
> As I read it, passing requires evidence that <X> and <Y> are equivalent, but the
> check is <a difference test / a point threshold>. If that's right, then a weaker or
> noisier measurement makes the criterion easier to pass rather than harder, so the
> verdict depends on the power of the check as much as on the data.
>
> I'm not claiming this makes your results wrong. It's a statement about what the
> criterion can establish, not about whether the thing it asserts is true.
>
> I hit exactly this in my own benchmark. My matched-proxy check was
> `abs(mean_auroc - 0.5) < 0.10` at four seeds a side, which passed a genuinely
> leaking case 43.5% of the time. The fix was an equivalence test at the margin the
> criterion already implied: the published claim didn't change, the sample size did.
> My own coding, including the two criteria where I failed this, is at
> <link to audit/SELF_CODING.md>.
>
> Two questions:
>
> 1. Have I coded this wrong? In particular, is `<criterion>` making a claim I've
>    misread as an equivalence?
> 2. Is there a power analysis, a rejection log, or an audit that shows this check
>    returning a negative? If so I'd like to cite it, and the row comes out.
>
> I'm planning to publish the coding on <date, ≥4 weeks out>, with any corrections
> and responses printed next to the rows. If you'd like more time, say so and you'll
> have it.
>
> The protocol I'm following, including what I think this coding cannot conclude, is
> at <link to docs/SURVEY_PROTOCOL.md>.
>
> Thanks for reading,
> Aarav Shah

---

## Template: a criterion coded UNDEMONSTRATED

This is the milder one and needs to say so in the first line, or it reads as an
accusation it isn't.

> Subject: Is there a published rejection for `<criterion>` in `<benchmark>`?
>
> Dear <names>,
>
> Short question, and a mild one. I'm coding published benchmarks for whether their
> preconditions come with evidence that they can return a negative — a power curve, a
> probe the check rejects, or a rejection reported alongside the passes.
>
> For `<criterion>` at `<source>` I couldn't find one. That's very likely me missing
> it, or it living somewhere I didn't look, which is why I'm asking rather than
> publishing.
>
> To be clear about what this is and isn't: "nothing published shows it failing" is
> not "it can't fail". Most criteria I've coded are in this bucket, including four of
> my own six. It's usually a reporting gap rather than a design one.
>
> Is there a rejection or a power analysis I've missed? If so I'll cite it and drop
> the row.
>
> Thanks,
> Aarav Shah

---

## Handling replies

**"You've coded it wrong."** Likeliest and most useful outcome. Change the row, note
who corrected it and when, and thank them. If you disagree, the row stays with *both*
readings printed — yours and theirs — not yours alone.

**"Here's the power analysis."** The row comes out. Cite what they sent.

**"Why are you doing this?"** Answer plainly: you found the defect in your own
benchmark, wanted to know whether it was a general pattern, pre-registered a survey
to find out, and it did not generalize in the form you first proposed. That is in the
paper as a withdrawn claim. What's left is narrower.

**Anger.** Take the substance, drop the rest, and answer only the substance. If they
ask you not to include their benchmark: you are not obliged to agree, but you should
say clearly that the row will name the criterion and not them, publish their
objection next to it, and mean it.

**Silence.** Publish after the window. Note that they were contacted on <date> and
did not reply — a neutral fact, and it also protects them: a reader can see the
correction channel was open.

---

## Do not

- Send the comparative table to anyone before publication.
- Publish a benchmark whose authors were never contacted.
- Describe an `UNDEMONSTRATED` row as a check that cannot fail.
- Post about the survey publicly before the window closes. Everyone named should read
  it in an email from you first, not in a thread.
- Quote "0 of 77" in any of it. The denominator that survives contact with the people
  being named is inverted-over-equivalence-shaped; see `SURVEY_PROTOCOL.md` §4.
