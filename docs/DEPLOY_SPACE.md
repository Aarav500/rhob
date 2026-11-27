# Deploying the Leaderboard to a Hugging Face Space

The Gradio app in [`space/`](../space/) runs locally with no setup beyond
`pip install -e ".[space]"`. Making it publicly live on Hugging Face Spaces needs a
one-time manual setup, since it requires your HF account and a write token that only
you can create.

## One-time setup

1. **Create the Space.**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Owner: your account or org (e.g. `Aarav500`).
   - Space name: e.g. `rhob-leaderboard`.
   - SDK: **Gradio**.
   - Visibility: public (or private, if you'd rather not expose it yet).
   - You don't need to upload anything yet -- the deploy workflow does that.

2. **Create a write token.**
   - [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) -> New
     token -> role **Write**.
   - Copy it; you won't see it again.

3. **Add the token as a GitHub repo secret.**
   - This repo's Settings -> Secrets and variables -> Actions -> New repository secret.
   - Name: `HF_TOKEN`. Value: the token from step 2.

## Deploying

Once the secret is set, trigger the deploy workflow manually (it's
`workflow_dispatch`-only, not automatic on every push, so it never fails loudly before
you've done the setup above):

```bash
gh workflow run deploy_space.yml -f space_repo="Aarav500/rhob-leaderboard"
```

Or from the GitHub UI: Actions -> "deploy-space" -> Run workflow -> enter the Space repo
ID (e.g. `Aarav500/rhob-leaderboard`).

This copies `space/app.py`, `space/requirements.txt`, `space/README.md` (the HF Space
metadata card), a snapshot of the `leaderboard/` directory, and
`docs/figures/sign_randomization_impact.json` (into `figures/`) into the Space repo. The
transfer panel reads `leaderboard/cross_family_transfer_v5.json` when the snapshot has
it, and otherwise that figures file. The Space then installs `rhob` itself from this
repository, so `app.py` can use `rhob.v3.leaderboard` normally, exactly as it does when
run locally.

The installed `rhob` is the commit you deployed from. `space/requirements.txt` in the
repository names `main`, so a local `pip install -r space/requirements.txt` tracks
`main`; the workflow rewrites that line in its staged copy to the deploying commit
(`rhob @ git+https://github.com/<repo>.git@<sha>`), so the Space's app and its `rhob`
are always one commit, however far `main` has moved when the Space next rebuilds.
Deploying from a branch other than `main` is allowed, with a warning in the run log: the
Space then depends on that commit staying reachable on GitHub. The workflow also checks
that `space_repo` has the form `OWNER/NAME` and that `HF_TOKEN` is set before it uploads
anything, and installs `huggingface_hub` 1.33.0, the last 1.x release, inside the
`space` extra's `<2` range.

## Keeping it up to date

Re-run the same `gh workflow run` command (or the UI equivalent) any time you want the
Space to pick up new leaderboard data or app changes. This is not automatic on every
push to `main` -- decide whether you want it to be, and if so, change
`.github/workflows/deploy_space.yml`'s trigger from `workflow_dispatch` to also include
`push: branches: [main]` once you're comfortable with it redeploying on every commit.

## Once it's live

Update the "Interactive Leaderboard" link in [README.md](../README.md) to point at
`https://huggingface.co/spaces/<space_repo>` instead of "(coming soon)".
