# Security Policy

## Scope

RHOB is a research benchmark: a Python library, a set of simulated
environments, and detector baselines. It does not run as a hosted service and
does not process user data. Security reports here are most relevant to things
like:

- Arbitrary code execution via crafted input files (e.g. leaderboard JSON,
  HDF5 datasets, or config files loaded by the harness)
- Dependency vulnerabilities in `pyproject.toml` that affect this package's
  usage
- Supply-chain issues in the release/publishing process

It is **not** the right channel for methodology disagreements or "this
detector's numbers look wrong" — use a
[Benchmark Feedback issue](https://github.com/Aarav500/rhob/issues/new?template=benchmark_feedback.md)
for those instead.

## Supported Versions

Only the latest tagged release is actively supported. Older tags (`v1.0`,
`v1.1`, ...) are kept for reproducibility of published results but do not
receive security fixes.

| Version | Supported |
| --- | --- |
| Latest tag on `main` | Yes |
| Older tags | No |

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security reports. Instead,
email **aarav7.shah@gmail.com** with:

- A description of the issue and its potential impact
- Steps to reproduce (a minimal script or input file, if applicable)
- Any suggested fix, if you have one

You should expect an initial response within 5 business days. If the issue is
confirmed, we'll work with you on a fix and coordinate disclosure timing before
any public write-up.
