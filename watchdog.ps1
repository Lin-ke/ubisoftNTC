# watchdog.ps1 — 周期性检查 CodeMaker session, 闲置超阈值就发"继续"
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File watchdog.ps1
#   (或编辑下方 CONFIG 区改 SessionId)
#
# 行为:
#   1. 每隔 IntervalSec 跑一次 `codemaker session list`, 解析目标 session 的
#      Updated 时间戳。
#   2. 闲置 (now - Updated) 超过 IdleMinutes 就发 "继续"。
#   3. 同一 session 两次发送之间至少间隔 CooldownMinutes, 避免狂轰滥炸。

# ============================== CONFIG ==============================

$SessionId       = "ses_176ee98b3ffeVL2gcWWFY1zK9x"  # 目标 session
$Message         = "继续"
$IdleMinutes     = 5     # 闲置多少分钟视为停了
$CooldownMinutes = 8     # 上次发送后至少冷却多久才能再发
$IntervalSec     = 60    # 检查频率
$LogPath         = Join-Path $PSScriptRoot "watchdog.log"

# Busy 判据 — 任一满足即视为 agent/subagent 还在干活, 不发"继续"
$GpuUtilBusyPct  = 10    # GPU 利用率 >= 此值 = busy (训练时通常 > 50%)
$GpuMemBusyMiB   = 3000  # 显存 >= 此值 = busy (排除残留 CUDA context, 一般 1-2GB)
$RunLogIdleMin   = 3     # d:\ntc\run.log 在过去 N 分钟内有更新 = busy

# ====================================================================

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogPath -Value $line
}

function Parse-UpdatedColumn($line) {
    # 输入 line 形如:
    #   "ses_xxx   Title              00:02"
    #   "ses_xxx   Title              23:56 · 2026/6/2"
    # 末尾匹配 (HH:MM)(可选 · YYYY/M/D)
    if ($line -match '(\d{1,2}:\d{2})(?:\s*·\s*(\d{4})/(\d{1,2})/(\d{1,2}))?\s*$') {
        $hm   = $matches[1]
        $h, $m = $hm -split ':'
        $now  = Get-Date
        if ($matches[2]) {
            return Get-Date -Year ([int]$matches[2]) -Month ([int]$matches[3]) `
                            -Day ([int]$matches[4]) -Hour ([int]$h) `
                            -Minute ([int]$m) -Second 0
        } else {
            # 无日期 = "今天"。但若解析出的时间晚于当前, 实际是昨天。
            $candidate = Get-Date -Year $now.Year -Month $now.Month -Day $now.Day `
                                  -Hour ([int]$h) -Minute ([int]$m) -Second 0
            if ($candidate -gt $now) {
                $candidate = $candidate.AddDays(-1)
            }
            return $candidate
        }
    }
    return $null
}

function Get-SessionUpdated($sid) {
    $output = & codemaker session list 2>&1
    $line = $output | Where-Object { $_ -match [regex]::Escape($sid) } | Select-Object -First 1
    if (-not $line) {
        Log "WARN: session $sid not found in list"
        return $null
    }
    $ts = Parse-UpdatedColumn $line
    if (-not $ts) {
        Log "WARN: could not parse Updated from line: $line"
    }
    return $ts
}

function Send-Continue($sid, $msg) {
    Log "SEND -> $sid : '$msg'"
    try {
        & codemaker run -s $sid $msg 2>&1 | Out-Null
        Log "SEND OK"
    } catch {
        Log "SEND FAILED: $_"
    }
}

function Test-GpuBusy {
    # 返回 $true 若 GPU 在用 (训练进行中)
    try {
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (-not $line) { return $false }
        $parts = $line -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 2) { return $false }
        $util = [int]$parts[0]
        $mem  = [int]$parts[1]
        if ($util -ge $GpuUtilBusyPct -or $mem -ge $GpuMemBusyMiB) {
            Log ("  GPU busy: util={0}% mem={1}MiB" -f $util, $mem)
            return $true
        }
    } catch {
        Log "  GPU check failed: $_"
    }
    return $false
}

function Test-RunLogBusy {
    # 返回 $true 若 run.log 在过去 RunLogIdleMin 分钟内被写过
    $p = Join-Path $PSScriptRoot "run.log"
    if (-not (Test-Path $p)) { return $false }
    $f = Get-Item $p
    if ($f.Length -eq 0) { return $false }
    $ageMin = ((Get-Date) - $f.LastWriteTime).TotalMinutes
    if ($ageMin -le $RunLogIdleMin) {
        Log ("  run.log busy: last-write {0:N1} min ago" -f $ageMin)
        return $true
    }
    return $false
}

# 主循环

Log "watchdog start  session=$SessionId  idle>=${IdleMinutes}min  cooldown=${CooldownMinutes}min  interval=${IntervalSec}s"

$lastSent = [DateTime]::MinValue

while ($true) {
    $busy = (Test-GpuBusy) -or (Test-RunLogBusy)

    if ($busy) {
        Log "skip: training/subagent still busy"
    } else {
        $updated = Get-SessionUpdated $SessionId
        if ($updated) {
            $idleMin = ((Get-Date) - $updated).TotalMinutes
            $sinceSentMin = ((Get-Date) - $lastSent).TotalMinutes

            if ($idleMin -ge $IdleMinutes -and $sinceSentMin -ge $CooldownMinutes) {
                Log ("idle {0:N1} min, GPU/log free, sending continue" -f $idleMin)
                Send-Continue $SessionId $Message
                $lastSent = Get-Date
            } else {
                Log ("idle {0:N1} min  (threshold {1}, since-sent {2:N1} min)" `
                     -f $idleMin, $IdleMinutes, $sinceSentMin)
            }
        }
    }
    Start-Sleep -Seconds $IntervalSec
}
