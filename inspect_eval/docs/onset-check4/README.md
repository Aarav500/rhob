# Cross-check 4 inputs: the reviewers' candidate reads, judged blind

Step 2 of the order of work in [onset-protocol.md](../onset-protocol.md), done on 2026-10-02
before `onset_from_logs.py` was run on any trial.

- `candidates.json`, `agent_lists.json` and `watched_files.json` are the output of
  `scripts/reviewer_candidates.py` (commit `26ea61d`), run once on the relabel label files.
  `agent_lists.json` is what the separate agents saw: each reviewer's entries that name a
  watched file, in order, with labels, kinds and the reviewers' free-text reasons removed.
- `agent_prompt.txt` is the agents' exact prompt.
- `agent_input_manifest.json` lists every file in the agents' working folder, with its SHA-256:
  copies of the transcripts (Harbor trials as their `agent/` folder only, Inspect samples as
  their messages only), the lists in 17 shards, the watched files and the definition.
- `agent_raw_output.json` is what the agents returned, unedited.
- `../onset-reviewer-reads.json` is the merged list the onset script reads: per reviewer per
  trial, the reviewer read (or none) and each rejected candidate with its reason.

**The agents.** Model `claude-opus-5-5`, run as Claude Code workflow subagents at high effort,
one per shard (16), each with only the prompt above. None was given the protocol, the read
rule, the onset script or anything it produced. They were told to work only in a folder that
held copies of their inputs; that was an instruction, not a technical barrier.

**One repair.** In the first pass, five trials came back with one reviewer's list returned
twice and the other's missing (P-glm polyglot-rust-c, P-glm sparql-university, P-kimi
qemu-startup, C1 regex-chess, C2 rstan-to-pystan). Because the duplicate could not be
assigned with certainty, both reviewers' lists of those five trials were judged again by one
fresh agent with the same prompt and a seventeenth shard (`redo_shard_16` in the raw output),
and those ten results replace the first pass's.

**Result of the merge.** All 928 reviewer lists with a candidate were judged; 526 end in a
read. The other 486 reviewer lists of the labelled trials had no candidate.
