<#
.SYNOPSIS
  Launch leaderboard replicates in parallel, one process per replicate.

.DESCRIPTION
  Each replicate is an independent draw of the whole benchmark and shares nothing with
  the others, so R replicates on R cores cost the wall-clock of one. Processes run at
  BelowNormal priority so a long replication wave does not make the machine unusable.

  Memory, not CPU, is the binding constraint: each process holds a rollout cache for
  every (family, difficulty) cell it has evaluated, and the cache peaks once the first
  detector has walked the whole suite. Size -Count against available RAM, not core count.

.EXAMPLE
  powershell -File scripts/run_replicates.ps1 -Start 0 -Count 4
#>
param(
    [int]$Start = 0,
    [int]$Count = 4,
    [int]$NSeeds = 5,
    [string]$OutDir = "results/replication"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$logDir = Join-Path $repo "logs\replication"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repo $OutDir) | Out-Null

$launched = @()
for ($i = $Start; $i -lt ($Start + $Count); $i++) {
    $log = Join-Path $logDir "rep_$i.log"
    $err = Join-Path $logDir "rep_$i.err"
    $p = Start-Process -FilePath "python" `
        -ArgumentList "scripts/replicate_leaderboard.py", "--replicate-id", "$i", `
                      "--n-seeds", "$NSeeds", "--out-dir", "$OutDir" `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $log -RedirectStandardError $err `
        -PassThru -WindowStyle Hidden
    try { $p.PriorityClass = "BelowNormal" } catch { }
    $launched += [pscustomobject]@{ Replicate = $i; Pid = $p.Id }
}

$launched | Format-Table -AutoSize
Write-Output ("launched {0} replicates ({1}..{2})" -f $Count, $Start, ($Start + $Count - 1))
