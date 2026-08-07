# Re-run the replication on the PINNED Windows stack.
#
# The published 20 draws ran on Amazon Linux against pip-resolved version RANGES,
# because requirements-lock.txt was generated on win32 and does not install there.
# That left three measurement bases in one paper: the ledger and the
# sign-randomization artifacts on the pinned stack, the replication on another.
# The lock's native platform is this one, so the reconciling run belongs here.
#
# Threads are pinned to 1 per process. The default is one BLAS thread per core PER
# PROCESS; at 20 processes that was 1561 threads on 32 cores and produced zero
# completed detectors in 8.75 hours of pure context-switching.
param([int]$Concurrency = 3, [int]$Total = 20, [string]$OutDir = "results/replication_pinned")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"; $env:VECLIB_MAXIMUM_THREADS = "1"
$env:RHOB_SOURCE_COMMIT = (git -C $repo rev-parse HEAD)
$env:RHOB_SOURCE_BRANCH = (git -C $repo rev-parse --abbrev-ref HEAD)
$logDir = Join-Path $repo "logs\replication_pinned"
New-Item -ItemType Directory -Force -Path $logDir, (Join-Path $repo $OutDir) | Out-Null
$running = @()
for ($i = 0; $i -lt $Total; $i++) {
  while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Concurrency) { Start-Sleep -Seconds 20 }
  $p = Start-Process -FilePath "python" `
    -ArgumentList "scripts/replicate_leaderboard.py","--replicate-id","$i","--n-seeds","5","--out-dir","$OutDir" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $logDir "rep_$i.log") `
    -RedirectStandardError  (Join-Path $logDir "rep_$i.err") `
    -PassThru -WindowStyle Hidden
  try { $p.PriorityClass = "BelowNormal" } catch { }
  $running += $p
  Write-Output "launched replicate $i (pid $($p.Id))"
}
Write-Output "all $Total launched; concurrency $Concurrency"
