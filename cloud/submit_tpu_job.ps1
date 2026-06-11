<#
.SYNOPSIS
Local Google TPU VM orchestrator for the GRPO baseline workflow.

.EXAMPLE
.\cloud\submit_tpu_job.ps1 preflight -DryRun
.\cloud\submit_tpu_job.ps1 ensure-tpu
.\cloud\submit_tpu_job.ps1 bootstrap -RunId setup-001
.\cloud\submit_tpu_job.ps1 submit-baseline -RunId baseline-001
.\cloud\submit_tpu_job.ps1 submit-reward-sweep -RunId reward-grid-001
.\cloud\submit_tpu_job.ps1 status-sweep -RunId reward-grid-001
.\cloud\submit_tpu_job.ps1 fetch-sweep -RunId reward-grid-001
.\cloud\submit_tpu_job.ps1 submit-reward-continuation -RunId reward-continuation-001
.\cloud\submit_tpu_job.ps1 status-continuation -RunId reward-continuation-001
.\cloud\submit_tpu_job.ps1 fetch-continuation -RunId reward-continuation-001
.\cloud\submit_tpu_job.ps1 submit-candidate-eval -RunId candidate-eval-r3-r5-001
.\cloud\submit_tpu_job.ps1 status-candidate-eval -RunId candidate-eval-r3-r5-001
.\cloud\submit_tpu_job.ps1 fetch-candidate-eval -RunId candidate-eval-r3-r5-001
.\cloud\submit_tpu_job.ps1 submit-reward-dense -RunId reward-dense-001
.\cloud\submit_tpu_job.ps1 status-reward-dense -RunId reward-dense-001
.\cloud\submit_tpu_job.ps1 fetch-reward-dense -RunId reward-dense-001
.\cloud\submit_tpu_job.ps1 submit-r7-large-eval -RunId r7-large-eval-001
.\cloud\submit_tpu_job.ps1 status-r7-large-eval -RunId r7-large-eval-001
.\cloud\submit_tpu_job.ps1 fetch-r7-large-eval -RunId r7-large-eval-001
.\cloud\submit_tpu_job.ps1 submit-reward-r9 -RunId reward-r9-closed-answer-001
.\cloud\submit_tpu_job.ps1 status-reward-r9 -RunId reward-r9-closed-answer-001
.\cloud\submit_tpu_job.ps1 fetch-reward-r9 -RunId reward-r9-closed-answer-001
.\cloud\submit_tpu_job.ps1 submit-reward-r10 -RunId reward-r10-numeric-guarded-001
.\cloud\submit_tpu_job.ps1 submit-k8-pilot -RunId reward-k8-beta004-pilot-001
.\cloud\submit_tpu_job.ps1 submit-k8-public-beta -RunId reward-k8-public-beta-001
.\cloud\submit_tpu_job.ps1 submit-k8-r12-simple-full -RunId reward-k8-beta004-r12-full-001
.\cloud\submit_tpu_job.ps1 submit-reward-only-r12-full -RunId reward-only-r12-full-001
.\cloud\submit_tpu_job.ps1 submit-reward-only-r12-complete-from500 -RunId reward-only-r12-full-complete-001
.\cloud\submit_tpu_job.ps1 submit-r12-best-large-eval -RunId r12-best-large-eval-001
.\cloud\submit_tpu_job.ps1 submit-r12-non-r64-pilot -RunId r12-non-r64-pilot-001
.\cloud\submit_tpu_job.ps1 submit-r12-lora-public-tuning -RunId r12-lora-public-tuning-001
.\cloud\submit_tpu_job.ps1 submit-r12-r64-beta-clip-tuning -RunId r12-r64-beta-clip-tuning-001
.\cloud\submit_tpu_job.ps1 submit-r12-r64-small-beta-tuning -RunId r12-r64-small-beta-tuning-001
.\cloud\submit_tpu_job.ps1 submit-r12-tail-stability -RunId r12-tail-stability-001
.\cloud\submit_tpu_job.ps1 submit-r12-high-rank-pilot -RunId r12-high-rank-pilot-001
.\cloud\submit_tpu_job.ps1 submit-r12-high-rank-alpha64-only -RunId r12-high-rank-alpha64-001
.\cloud\submit_tpu_job.ps1 submit-r12-r64-lr-smoothing -RunId r12-r64-lr-smoothing-001
.\cloud\submit_tpu_job.ps1 eval-checkpoints -RunId baseline-001
.\cloud\submit_tpu_job.ps1 status -RunId baseline-001
.\cloud\submit_tpu_job.ps1 ensure-storage
.\cloud\submit_tpu_job.ps1 sync-storage -RunId baseline-001
.\cloud\submit_tpu_job.ps1 fetch -RunId baseline-001
.\cloud\submit_tpu_job.ps1 stop-tpu
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("preflight", "ensure-tpu", "ensure-storage", "bootstrap", "submit-baseline", "submit-reward-sweep", "submit-reward-continuation", "submit-candidate-eval", "submit-reward-dense", "submit-r7-large-eval", "submit-r12-best-large-eval", "submit-reward-r9", "submit-reward-r10", "submit-k8-pilot", "submit-k8-r10-only", "submit-k8-r11-fallback-only", "submit-k8-r12-simple-only", "submit-k8-r12-simple-full", "submit-reward-only-r12-full", "submit-reward-only-r12-complete-from500", "submit-r12-non-r64-pilot", "submit-r12-lora-public-tuning", "submit-r12-r64-beta-clip-tuning", "submit-r12-r64-small-beta-tuning", "submit-r12-tail-stability", "submit-r12-high-rank-pilot", "submit-r12-high-rank-alpha64-only", "submit-r12-r64-lr-smoothing", "submit-r12-public-strong-tuning", "submit-k8-public-beta", "submit-k8-r13-public-beta-only", "submit-k8-r14-public-beta-only", "eval-checkpoints", "status", "status-sweep", "status-continuation", "status-candidate-eval", "status-reward-dense", "status-r7-large-eval", "status-r12-best-large-eval", "status-reward-r9", "status-reward-r10", "status-k8-pilot", "resume-k8-pilot", "stop-reward-r10", "stop-k8-pilot", "fetch", "fetch-sweep", "fetch-continuation", "fetch-candidate-eval", "fetch-reward-dense", "fetch-r7-large-eval", "fetch-r12-best-large-eval", "fetch-reward-r9", "fetch-reward-r10", "fetch-k8-pilot", "sync-storage", "restore-cache", "start-tpu", "stop-tpu", "delete-tpu")]
    [string]$Command = "preflight",

    [string]$RunId = ("baseline-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [switch]$DryRun,
    [switch]$TinySmoke,
    [switch]$KeepBundle,
    [string]$ConfigPath = "",
    [string]$SecretsFileOverride
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $ScriptDir "tpu_config.local.ps1"
}
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

# Defaults match create_tpu_env.sh. Override by copying
# cloud/tpu_config.example.ps1 to cloud/tpu_config.local.ps1.
$ProjectId = "grpo-tpu-play-20260603-z9s5"
$TpuName = "grpo-play-v6e"
$Zone = "us-east5-b"
$Region = "us-east5"
$AcceleratorType = "v6e-1"
$RuntimeVersion = "v2-alpha-tpuv6e"
$UseIapTunnel = $true
$UseOpenSshIap = $false
$IapTargetName = ""
$SshUser = $env:USERNAME
$SshKeyPath = "$env:USERPROFILE\.ssh\google_compute_engine"
$SshKnownHostsPath = "$env:USERPROFILE\.ssh\google_compute_known_hosts"
$RemoteRoot = "~/tpu-runs"
$RemoteToolsDir = "~/tpu-runs/_tools"
$RemoteIncomingDir = "~/tpu-runs/_incoming"
$RemoteVenv = "~/venvs/tunix"
$SecretsFile = ".env"
$LocalArtifactsRoot = "artifacts/cloud"
$UseStorage = $true
$StorageBucket = ""
$StorageLocation = $Region
$StorageClass = "STANDARD"
$StoragePrefix = "tpu-runs"
$StorageCachePrefix = "cache"

if (Test-Path -LiteralPath $ConfigPath) {
    . $ConfigPath
}
if ($SecretsFileOverride) {
    $SecretsFile = $SecretsFileOverride
}
if (-not $UseStorage) {
    $StorageBucket = ""
}
elseif ([string]::IsNullOrWhiteSpace($StorageBucket)) {
    $StorageBucket = "$ProjectId-tpu-artifacts"
}

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Quote-Arg([string]$Arg) {
    if ($Arg -match '[\s"`$]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

function Format-Command([string]$Exe, [string[]]$ArgumentList) {
    return (@($Exe) + $ArgumentList | ForEach-Object { Quote-Arg $_ }) -join " "
}

function Invoke-External([string]$Exe, [string[]]$ArgumentList) {
    $display = Format-Command $Exe $ArgumentList
    if ($DryRun) {
        Write-Host "[dry-run] $display"
        return
    }
    Write-Host $display -ForegroundColor DarkGray
    & $Exe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $display"
    }
}

function Test-ExternalSuccess([string]$Exe, [string[]]$ArgumentList) {
    $display = Format-Command $Exe $ArgumentList
    if ($DryRun) {
        Write-Host "[dry-run] $display"
        return $false
    }
    try {
        & $Exe @ArgumentList *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Get-GCloudExecutable {
    $cmd = Get-Command "gcloud.cmd" -ErrorAction SilentlyContinue
    if (-not $cmd) {
        $cmd = Get-Command "gcloud" -ErrorAction SilentlyContinue
    }
    if (-not $cmd) {
        $default = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        if (Test-Path -LiteralPath $default) {
            return $default
        }
    }
    if (-not $cmd) {
        throw "gcloud is not installed or not on PATH. Install Google Cloud CLI, then run gcloud auth login."
    }
    return $cmd.Source
}

function Invoke-GCloud([string[]]$ArgumentList) {
    Invoke-External (Get-GCloudExecutable) $ArgumentList
}

function Test-GCloud([string[]]$ArgumentList) {
    return Test-ExternalSuccess (Get-GCloudExecutable) $ArgumentList
}

function Get-FreeTcpPort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), 0)
    $listener.Start()
    try {
        return $listener.LocalEndpoint.Port
    }
    finally {
        $listener.Stop()
    }
}

function Get-IapTarget {
    if (-not [string]::IsNullOrWhiteSpace($IapTargetName)) {
        return $IapTargetName
    }
    throw "OpenSSH IAP mode requires `$IapTargetName in cloud/tpu_config.local.ps1. Use the TPU VM hostname, e.g. t1v-...-w-0."
}

function Wait-LocalPort([int]$Port) {
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Milliseconds 500
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($iar.AsyncWaitHandle.WaitOne(250)) {
                $client.EndConnect($iar)
                return
            }
        }
        catch {
        }
        finally {
            $client.Close()
        }
    }
    throw "IAP tunnel did not open on local port $Port."
}

function Start-OpenSshIapTunnel {
    $target = Get-IapTarget
    $port = Get-FreeTcpPort
    $args = @(
        "alpha", "compute", "start-iap-tunnel", $target, "22",
        "--local-host-port=127.0.0.1:$port",
        "--project=$ProjectId",
        "--zone=$Zone"
    )
    $gcloud = Get-GCloudExecutable
    $display = Format-Command $gcloud $args
    if ($DryRun) {
        Write-Host "[dry-run] $display"
        return [PSCustomObject]@{ Port = $port; Target = $target }
    }
    Write-Host $display -ForegroundColor DarkGray
    $process = Start-Process -FilePath $gcloud -ArgumentList $args -WindowStyle Hidden -PassThru
    Wait-LocalPort $port
    return [PSCustomObject]@{ Port = $port; Target = $target; ProcessId = $process.Id }
}

function Stop-OpenSshIapTunnel($Tunnel) {
    if ($DryRun -or -not $Tunnel) {
        return
    }
    $portPattern = [regex]::Escape("--local-host-port=127.0.0.1:$($Tunnel.Port)")
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine -match "start-iap-tunnel" -and $_.CommandLine -match $portPattern } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

function Get-OpenSshArgs([int]$Port) {
    return @(
        "-4",
        "-p", "$Port",
        "-i", (Resolve-RepoPath $SshKeyPath),
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=$(Resolve-RepoPath $SshKnownHostsPath)",
        "-o", "ServerAliveInterval=30"
    )
}

function Convert-ScpEndpoint([string]$Endpoint) {
    $prefix = "$TpuName`:"
    if ($Endpoint.StartsWith($prefix)) {
        return "$SshUser@127.0.0.1:" + $Endpoint.Substring($prefix.Length)
    }
    return $Endpoint
}

function GCloudTpuBaseArgs([string]$Verb) {
    $cmdArgs = @("alpha", "compute", "tpus", "tpu-vm", $Verb)
    return $cmdArgs
}

function Add-AccessArgs([string[]]$ArgumentList) {
    $out = @($ArgumentList)
    $out += @("--project=$ProjectId", "--zone=$Zone")
    if ($UseIapTunnel) {
        $out += "--tunnel-through-iap"
    }
    return $out
}

function Invoke-Remote([string]$RemoteCommand) {
    if ($UseOpenSshIap) {
        $tunnel = Start-OpenSshIapTunnel
        try {
            $args = Get-OpenSshArgs $tunnel.Port
            $args += @("$SshUser@127.0.0.1", $RemoteCommand)
            Invoke-External "ssh.exe" $args
        }
        finally {
            Stop-OpenSshIapTunnel $tunnel
        }
        return
    }

    $cmdArgs = GCloudTpuBaseArgs "ssh"
    $cmdArgs += $TpuName
    $cmdArgs = Add-AccessArgs $cmdArgs
    $cmdArgs += "--command=$RemoteCommand"
    Invoke-GCloud $cmdArgs
}

function Invoke-RemoteScp([string]$Source, [string]$Destination, [switch]$Recurse) {
    if ($UseOpenSshIap) {
        $tunnel = Start-OpenSshIapTunnel
        try {
            $args = @("-P", "$($tunnel.Port)")
            if ($Recurse) {
                $args += "-r"
            }
            $args += @(
                "-i", (Resolve-RepoPath $SshKeyPath),
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=$(Resolve-RepoPath $SshKnownHostsPath)",
                "-o", "ServerAliveInterval=30",
                (Convert-ScpEndpoint $Source),
                (Convert-ScpEndpoint $Destination)
            )
            Invoke-External "scp.exe" $args
        }
        finally {
            Stop-OpenSshIapTunnel $tunnel
        }
        return
    }

    $cmdArgs = GCloudTpuBaseArgs "scp"
    if ($Recurse) {
        $cmdArgs += "--recurse"
    }
    $cmdArgs += @($Source, $Destination)
    $cmdArgs = Add-AccessArgs $cmdArgs
    Invoke-GCloud $cmdArgs
}

function Assert-GCloudAvailable {
    if ($DryRun) {
        return
    }
    if (-not (Get-Command "gcloud" -ErrorAction SilentlyContinue)) {
        throw "gcloud is not installed or not on PATH. Install Google Cloud CLI, then run gcloud auth login."
    }
}

function Assert-RunId {
    if ($RunId -notmatch '^[A-Za-z0-9._-]+$') {
        throw "RunId may only contain letters, numbers, dot, underscore, and dash."
    }
}

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $RepoRoot $Path)
}

function Get-RemotePath([string]$Base, [string]$Name) {
    return ($Base.TrimEnd("/") + "/" + $Name)
}

function New-PortableZipFromDirectory([string]$SourceDir, [string]$DestinationPath) {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $DestinationPath) {
        Remove-Item -LiteralPath $DestinationPath -Force
    }

    $sourceFull = [System.IO.Path]::GetFullPath($SourceDir).TrimEnd("\", "/")
    $archive = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceFull -Recurse -File | ForEach-Object {
            $relative = $_.FullName.Substring($sourceFull.Length).TrimStart("\", "/")
            $entryName = $relative.Replace("\", "/")
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $archive.Dispose()
    }
}

function New-CodeBundle {
    Assert-RunId
    Write-Step "Packaging current working tree"
    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tpu-job-" + $RunId + "-" + [Guid]::NewGuid().ToString("N"))
    $stage = Join-Path $tempRoot "bundle"
    $srcStage = Join-Path $stage "src"
    $metaStage = Join-Path $stage "meta"
    New-Item -ItemType Directory -Path $srcStage, $metaStage -Force | Out-Null

    $files = & git -C $RepoRoot ls-files --cached --modified --others --exclude-standard
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed; run this from a git checkout."
    }

    foreach ($rel in ($files | Sort-Object -Unique)) {
        if ([string]::IsNullOrWhiteSpace($rel)) {
            continue
        }
        if ($rel -eq "artifacts" -or $rel -like "artifacts/*") {
            continue
        }
        if ($rel -like "*.pyc" -or $rel -like "*/__pycache__/*") {
            continue
        }
        $source = Join-Path $RepoRoot $rel
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            continue
        }
        $dest = Join-Path $srcStage $rel
        $destDir = Split-Path -Parent $dest
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $dest -Force
    }

    (& git -C $RepoRoot rev-parse HEAD) | Set-Content -Path (Join-Path $metaStage "git_commit.txt") -Encoding utf8
    (& git -C $RepoRoot status --short --branch) | Set-Content -Path (Join-Path $metaStage "git_status.txt") -Encoding utf8
    (& git -C $RepoRoot diff --binary) | Set-Content -Path (Join-Path $metaStage "dirty.patch") -Encoding utf8
    (& git -C $RepoRoot diff --cached --binary) | Set-Content -Path (Join-Path $metaStage "staged.patch") -Encoding utf8

    $zip = Join-Path $tempRoot "$RunId.zip"
    New-PortableZipFromDirectory -SourceDir $stage -DestinationPath $zip
    Write-Host "Bundle: $zip"
    return [PSCustomObject]@{
        TempRoot = $tempRoot
        Zip = $zip
    }
}

function Remove-CodeBundle($Bundle) {
    if ($KeepBundle -or -not $Bundle) {
        return
    }
    if ($Bundle.TempRoot -and (Test-Path -LiteralPath $Bundle.TempRoot)) {
        Remove-Item -LiteralPath $Bundle.TempRoot -Recurse -Force
    }
}

function Upload-Runner {
    Write-Step "Uploading remote runner"
    Invoke-Remote "mkdir -p $RemoteToolsDir $RemoteIncomingDir"
    $runner = Join-Path $RepoRoot "cloud/remote_tpu_runner.sh"
    $remoteRunner = Get-RemotePath $RemoteToolsDir "remote_tpu_runner.sh"
    Invoke-RemoteScp $runner "${TpuName}:$remoteRunner"
    Invoke-Remote "chmod +x $remoteRunner"
    return $remoteRunner
}

function Upload-Bundle($Bundle) {
    Write-Step "Uploading code bundle"
    $remoteBundle = Get-RemotePath $RemoteIncomingDir "$RunId.zip"
    Invoke-RemoteScp $Bundle.Zip "${TpuName}:$remoteBundle"
    return $remoteBundle
}

function Upload-SecretsIfPresent {
    $localSecrets = Resolve-RepoPath $SecretsFile
    if (-not (Test-Path -LiteralPath $localSecrets -PathType Leaf)) {
        Write-Warning "Secrets file not found at $localSecrets. Continuing without uploading .env."
        return ""
    }
    Write-Step "Uploading secrets file"
    $remoteSecrets = Get-RemotePath $RemoteIncomingDir "$RunId.env"
    Invoke-RemoteScp $localSecrets "${TpuName}:$remoteSecrets"
    Invoke-Remote "chmod 600 $remoteSecrets"
    return $remoteSecrets
}

function Invoke-RemoteRunner([string]$RemoteRunner, [string]$RunnerCommand, [string]$RemoteBundle = "", [string]$RemoteSecrets = "") {
    $cmd = "bash $RemoteRunner $RunnerCommand --run-id $RunId --remote-root $RemoteRoot --venv $RemoteVenv"
    $cmd += " --project-id $ProjectId --storage-prefix $StoragePrefix --storage-cache-prefix $StorageCachePrefix"
    if (-not [string]::IsNullOrWhiteSpace($StorageBucket)) {
        $cmd += " --storage-bucket $StorageBucket"
    }
    if ($RemoteBundle) {
        $cmd += " --bundle $RemoteBundle"
    }
    if ($RemoteSecrets) {
        $cmd += " --secrets $RemoteSecrets"
    }
    if ($TinySmoke) {
        $cmd += " --tiny-smoke"
    }
    Invoke-Remote $cmd
}

function Preflight {
    Write-Step "Checking local Google Cloud CLI and project access"
    Assert-GCloudAvailable
    Invoke-GCloud @("--version")
    Invoke-GCloud @("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
    Invoke-GCloud @("config", "get-value", "project")
    Invoke-GCloud @("services", "list", "--enabled", "--filter=name:tpu.googleapis.com", "--project=$ProjectId")
    Invoke-GCloud @("compute", "tpus", "tpu-vm", "list", "--project=$ProjectId", "--zone=$Zone")
}

function Ensure-Tpu {
    Write-Step "Ensuring network prerequisites"
    Invoke-GCloud @(
        "compute", "networks", "subnets", "update", "default",
        "--region=$Region",
        "--enable-private-ip-google-access",
        "--project=$ProjectId"
    )

    if (-not (Test-GCloud @("compute", "routers", "describe", "nat-router", "--network=default", "--region=$Region", "--project=$ProjectId"))) {
        Invoke-GCloud @("compute", "routers", "create", "nat-router", "--network=default", "--region=$Region", "--project=$ProjectId")
    } else {
        Write-Host "Router nat-router already exists."
    }

    if (-not (Test-GCloud @("compute", "routers", "nats", "describe", "nat-config", "--router=nat-router", "--region=$Region", "--project=$ProjectId"))) {
        Invoke-GCloud @(
            "compute", "routers", "nats", "create", "nat-config",
            "--router=nat-router",
            "--region=$Region",
            "--auto-allocate-nat-external-ips",
            "--nat-all-subnet-ip-ranges",
            "--project=$ProjectId"
        )
    } else {
        Write-Host "Cloud NAT nat-config already exists."
    }

    if (-not (Test-GCloud @("compute", "firewall-rules", "describe", "allow-iap-ssh", "--project=$ProjectId"))) {
        Invoke-GCloud @(
            "compute", "firewall-rules", "create", "allow-iap-ssh",
            "--project=$ProjectId",
            "--network=default",
            "--source-ranges=35.235.240.0/20",
            "--allow=tcp:22"
        )
    } else {
        Write-Host "Firewall rule allow-iap-ssh already exists."
    }

    Write-Step "Ensuring TPU VM exists"
    if (Test-GCloud @("compute", "tpus", "tpu-vm", "describe", $TpuName, "--project=$ProjectId", "--zone=$Zone")) {
        Write-Host "TPU VM $TpuName already exists."
        return
    }

    Invoke-GCloud @(
        "compute", "tpus", "tpu-vm", "create", $TpuName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--accelerator-type=$AcceleratorType",
        "--version=$RuntimeVersion",
        "--internal-ips"
    )
}

function Get-StorageBucketUri {
    if ([string]::IsNullOrWhiteSpace($StorageBucket)) {
        throw "StorageBucket is empty. Set `$StorageBucket in cloud/tpu_config.local.ps1."
    }
    return "gs://$StorageBucket"
}

function Get-StorageRunUri {
    Assert-RunId
    return "$(Get-StorageBucketUri)/$StoragePrefix/$RunId"
}

function Ensure-Storage {
    Write-Step "Ensuring Cloud Storage bucket exists"
    Invoke-GCloud @("services", "enable", "storage.googleapis.com", "--project=$ProjectId")

    $bucketUri = Get-StorageBucketUri
    if (Test-GCloud @("storage", "buckets", "describe", $bucketUri, "--project=$ProjectId")) {
        Write-Host "Storage bucket $bucketUri already exists."
        return
    }

    Invoke-GCloud @(
        "storage", "buckets", "create", $bucketUri,
        "--project=$ProjectId",
        "--location=$StorageLocation",
        "--default-storage-class=$StorageClass",
        "--uniform-bucket-level-access"
    )
}

function Bootstrap-Remote {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "bootstrap" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-Baseline {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-baseline" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardSweep {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-sweep" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardContinuation {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-continuation" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-CandidateEval {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-candidate-eval" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardDense {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-dense" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R7LargeEval {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r7-large-eval" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12BestLargeEval {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-best-large-eval" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardR9 {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-r9" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardR10 {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-r10" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8Pilot {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-pilot" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R10Only {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r10-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R11FallbackOnly {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r11-fallback-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R12SimpleOnly {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r12-simple-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R12SimpleFull {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r12-simple-full" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardOnlyR12Full {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-only-r12-full" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-RewardOnlyR12CompleteFrom500 {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-reward-only-r12-complete-from500" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12NonR64Pilot {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-non-r64-pilot" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12LoraPublicTuning {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-lora-public-tuning" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12R64BetaClipTuning {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-r64-beta-clip-tuning" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12R64SmallBetaTuning {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-r64-small-beta-tuning" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12TailStability {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-tail-stability" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12HighRankPilot {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-high-rank-pilot" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12HighRankAlpha64Only {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-high-rank-alpha64-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12R64LrSmoothing {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-r64-lr-smoothing" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-R12PublicStrongTuning {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-r12-public-strong-tuning" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8PublicBeta {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-public-beta" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R13PublicBetaOnly {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r13-public-beta-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Submit-K8R14PublicBetaOnly {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "submit-k8-r14-public-beta-only" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Eval-Checkpoints {
    Assert-RunId
    $bundle = New-CodeBundle
    try {
        $runner = Upload-Runner
        $remoteBundle = Upload-Bundle $bundle
        $remoteSecrets = Upload-SecretsIfPresent
        Invoke-RemoteRunner $runner "eval-checkpoints" $remoteBundle $remoteSecrets
    } finally {
        Remove-CodeBundle $bundle
    }
}

function Status-Run {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status"
}

function Status-Sweep {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-sweep"
}

function Status-Continuation {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-continuation"
}

function Status-CandidateEval {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-candidate-eval"
}

function Status-RewardDense {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-reward-dense"
}

function Status-R7LargeEval {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-r7-large-eval"
}

function Status-R12BestLargeEval {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-r12-best-large-eval"
}

function Status-RewardR9 {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-reward-r9"
}

function Status-RewardR10 {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-reward-r10"
}

function Status-K8Pilot {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "status-k8-pilot"
}

function Resume-K8Pilot {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "resume-k8-pilot"
}

function Stop-RewardR10 {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "stop-reward-r10"
}

function Stop-K8Pilot {
    Assert-RunId
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "stop-k8-pilot"
}

function Fetch-Run {
    Assert-RunId
    Write-Step "Preparing remote result archive"
    $remoteRunDir = Get-RemotePath $RemoteRoot $RunId
    $remoteArchive = Get-RemotePath $remoteRunDir "$RunId-results.tar.gz"
    $remoteCommand = @"
RUN_DIR=$remoteRunDir
FETCH=`$RUN_DIR/fetch
ARCHIVE=$remoteArchive
rm -rf "`$FETCH" "`$ARCHIVE"
mkdir -p "`$FETCH"
for path in artifacts meta pipeline.log run_baseline.sh run_eval_checkpoints.sh run_reward_sweep.sh run_reward_continuation.sh run_candidate_eval.sh run_reward_dense.sh run_r7_large_eval.sh run_reward_r9.sh tensorboard runs; do
  if [ -e "`$RUN_DIR/`$path" ]; then
    cp -r "`$RUN_DIR/`$path" "`$FETCH/"
  fi
done
tar -czf "`$ARCHIVE" -C "`$FETCH" .
"@
    Invoke-Remote $remoteCommand

    $localRoot = Resolve-RepoPath $LocalArtifactsRoot
    $localDest = Join-Path $localRoot $RunId
    $localArchive = Join-Path $localDest "$RunId-results.tar.gz"

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $localDest -Force | Out-Null
    }

    Write-Step "Downloading result archive"
    Invoke-RemoteScp "${TpuName}:$remoteArchive" $localArchive

    if (-not $DryRun) {
        Write-Step "Extracting result archive"
        if (Get-Command "tar" -ErrorAction SilentlyContinue) {
            & tar -xzf $localArchive -C $localDest
            if ($LASTEXITCODE -ne 0) {
                throw "tar extraction failed: $localArchive"
            }
        } else {
            Write-Warning "tar is not available locally. Archive left at $localArchive"
        }
        Write-Host "Fetched run outputs to $localDest"
    }
}

function Fetch-RewardDense {
    Assert-RunId
    Write-Step "Preparing split reward result archives"
    $remoteRunDir = Get-RemotePath $RemoteRoot $RunId
    $remoteAnalysisArchive = Get-RemotePath $remoteRunDir "$RunId-analysis.tar.gz"
    $remoteArchiveList = Get-RemotePath $remoteRunDir "checkpoint_archives.txt"
    $remoteCommand = @"
RUN_DIR=$remoteRunDir
FETCH=`$RUN_DIR/fetch_analysis
ANALYSIS_ARCHIVE=$remoteAnalysisArchive
ARCHIVE_LIST=$remoteArchiveList
rm -rf "`$FETCH" "`$ANALYSIS_ARCHIVE" "`$ARCHIVE_LIST" "`$RUN_DIR"/$RunId-checkpoint-*.tar.gz "`$RUN_DIR"/$RunId-checkpoints-*.tar.gz
mkdir -p "`$FETCH/runs"
for path in artifacts meta pipeline.log run_baseline.sh run_eval_checkpoints.sh run_reward_sweep.sh run_reward_continuation.sh run_candidate_eval.sh run_reward_dense.sh run_r7_large_eval.sh run_reward_r9.sh run_reward_r10.sh run_k8_pilot.sh tensorboard; do
  if [ -e "`$RUN_DIR/`$path" ]; then
    cp -r "`$RUN_DIR/`$path" "`$FETCH/"
  fi
done
if [ -d "`$RUN_DIR/runs" ]; then
  for child in "`$RUN_DIR"/runs/*; do
    [ -d "`$child" ] || continue
    name=`$(basename "`$child")
    mkdir -p "`$FETCH/runs/`$name"
    for path in artifacts meta tensorboard train.log reward_mode.txt run_env.txt run_manifest.json extension_decision.json checkpoint_eval_plan.txt; do
      if [ -e "`$child/`$path" ]; then
        cp -r "`$child/`$path" "`$FETCH/runs/`$name/"
      fi
    done
  done
fi
tar -czf "`$ANALYSIS_ARCHIVE" -C "`$FETCH" .
: > "`$ARCHIVE_LIST"
if [ -d "`$RUN_DIR/runs" ]; then
  for child in "`$RUN_DIR"/runs/*; do
    [ -d "`$child" ] || continue
    name=`$(basename "`$child")
    if [ -d "`$child/ckpts/actor" ]; then
      find "`$child/ckpts/actor" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' |
        sed 's#.*/##' |
        sort -n |
        while read -r step; do
          printf '%s %s\n' "`$name" "`$step"
        done >> "`$ARCHIVE_LIST"
    fi
  done
fi
"@
    Invoke-Remote $remoteCommand

    $localRoot = Resolve-RepoPath $LocalArtifactsRoot
    $localDest = Join-Path $localRoot $RunId
    $localAnalysisArchive = Join-Path $localDest "$RunId-analysis.tar.gz"
    $localArchiveList = Join-Path $localDest "checkpoint_archives.txt"
    $localCheckpointArchiveDir = Join-Path $localDest "checkpoint_archives"

    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $localDest, $localCheckpointArchiveDir -Force | Out-Null
    }

    Write-Step "Downloading reward analysis archive"
    Invoke-RemoteScp "${TpuName}:$remoteAnalysisArchive" $localAnalysisArchive

    if (-not $DryRun) {
        Write-Step "Extracting reward analysis archive"
        if (Get-Command "tar" -ErrorAction SilentlyContinue) {
            & tar -xzf $localAnalysisArchive -C $localDest
            if ($LASTEXITCODE -ne 0) {
                throw "tar extraction failed: $localAnalysisArchive"
            }
        } else {
            Write-Warning "tar is not available locally. Archive left at $localAnalysisArchive"
        }
    }

    Write-Step "Downloading checkpoint list"
    Invoke-RemoteScp "${TpuName}:$remoteArchiveList" $localArchiveList

    if ($DryRun) {
        return
    }

    if (-not (Test-Path -LiteralPath $localArchiveList -PathType Leaf)) {
        Write-Warning "No checkpoint archive list downloaded."
        return
    }

    $checkpointEntries = Get-Content -LiteralPath $localArchiveList | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    Set-Content -LiteralPath $localArchiveList -Value "" -Encoding utf8
    foreach ($entry in $checkpointEntries) {
        $parts = $entry.Trim() -split "\s+"
        if ($parts.Count -lt 2) {
            Write-Warning "Skipping malformed checkpoint entry: $entry"
            continue
        }
        $runName = $parts[0]
        $step = $parts[1]
        $fileName = "$RunId-checkpoint-$runName-$step.tar.gz"
        $remoteCheckpointArchive = Get-RemotePath $remoteRunDir $fileName
        $localCheckpointArchive = Join-Path $localCheckpointArchiveDir $fileName
        $remoteCheckpointCommand = @"
RUN_DIR=$remoteRunDir
RUN_NAME=$runName
STEP=$step
ARCHIVE=$remoteCheckpointArchive
CHILD="`$RUN_DIR/runs/`$RUN_NAME"
test -d "`$CHILD/ckpts/actor/`$STEP"
rm -f "`$ARCHIVE"
tar -czf "`$ARCHIVE" -C "`$CHILD/ckpts/actor" "`$STEP"
"@
        Write-Step "Preparing checkpoint archive $fileName"
        Invoke-Remote $remoteCheckpointCommand

        Write-Step "Downloading checkpoint archive $fileName"
        Invoke-RemoteScp "${TpuName}:$remoteCheckpointArchive" $localCheckpointArchive
        Invoke-Remote "rm -f $remoteCheckpointArchive"
        Add-Content -LiteralPath $localArchiveList -Value ("checkpoint_archives/$fileName") -Encoding utf8

        if (Get-Command "tar" -ErrorAction SilentlyContinue) {
            $runDest = Join-Path (Join-Path (Join-Path $localDest "runs") $runName) "ckpts\actor"
            New-Item -ItemType Directory -Path $runDest -Force | Out-Null
            & tar -xzf $localCheckpointArchive -C $runDest
            if ($LASTEXITCODE -ne 0) {
                throw "checkpoint extraction failed: $localCheckpointArchive"
            }
        }
    }

    Write-Host "Fetched reward outputs to $localDest"
}

function Sync-Storage {
    Assert-RunId
    Ensure-Storage

    Write-Step "Syncing remote run outputs and model cache to Cloud Storage"
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "sync-storage"
}

function Restore-Cache {
    Ensure-Storage

    Write-Step "Restoring Hugging Face model cache from Cloud Storage to TPU VM"
    $runner = Upload-Runner
    Invoke-RemoteRunner $runner "restore-cache"
}

function Start-Tpu {
    Write-Step "Starting TPU VM"
    Invoke-GCloud @(
        "compute", "tpus", "tpu-vm", "start", $TpuName,
        "--project=$ProjectId",
        "--zone=$Zone"
    )
}

function Stop-Tpu {
    Write-Step "Stopping TPU VM"
    Invoke-GCloud @(
        "compute", "tpus", "tpu-vm", "stop", $TpuName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--quiet"
    )
}

function Delete-Tpu {
    Write-Step "Deleting TPU VM"
    Invoke-GCloud @(
        "compute", "tpus", "tpu-vm", "delete", $TpuName,
        "--project=$ProjectId",
        "--zone=$Zone",
        "--quiet"
    )
}

switch ($Command) {
    "preflight" { Preflight }
    "ensure-tpu" { Ensure-Tpu }
    "ensure-storage" { Ensure-Storage }
    "bootstrap" { Bootstrap-Remote }
    "submit-baseline" { Submit-Baseline }
    "submit-reward-sweep" { Submit-RewardSweep }
    "submit-reward-continuation" { Submit-RewardContinuation }
    "submit-candidate-eval" { Submit-CandidateEval }
    "submit-reward-dense" { Submit-RewardDense }
    "submit-r7-large-eval" { Submit-R7LargeEval }
    "submit-r12-best-large-eval" { Submit-R12BestLargeEval }
    "submit-reward-r9" { Submit-RewardR9 }
    "submit-reward-r10" { Submit-RewardR10 }
    "submit-k8-pilot" { Submit-K8Pilot }
    "submit-k8-r10-only" { Submit-K8R10Only }
    "submit-k8-r11-fallback-only" { Submit-K8R11FallbackOnly }
    "submit-k8-r12-simple-only" { Submit-K8R12SimpleOnly }
    "submit-k8-r12-simple-full" { Submit-K8R12SimpleFull }
    "submit-reward-only-r12-full" { Submit-RewardOnlyR12Full }
    "submit-reward-only-r12-complete-from500" { Submit-RewardOnlyR12CompleteFrom500 }
    "submit-r12-non-r64-pilot" { Submit-R12NonR64Pilot }
    "submit-r12-lora-public-tuning" { Submit-R12LoraPublicTuning }
    "submit-r12-r64-beta-clip-tuning" { Submit-R12R64BetaClipTuning }
    "submit-r12-r64-small-beta-tuning" { Submit-R12R64SmallBetaTuning }
    "submit-r12-tail-stability" { Submit-R12TailStability }
    "submit-r12-high-rank-pilot" { Submit-R12HighRankPilot }
    "submit-r12-high-rank-alpha64-only" { Submit-R12HighRankAlpha64Only }
    "submit-r12-r64-lr-smoothing" { Submit-R12R64LrSmoothing }
    "submit-r12-public-strong-tuning" { Submit-R12PublicStrongTuning }
    "submit-k8-public-beta" { Submit-K8PublicBeta }
    "submit-k8-r13-public-beta-only" { Submit-K8R13PublicBetaOnly }
    "submit-k8-r14-public-beta-only" { Submit-K8R14PublicBetaOnly }
    "eval-checkpoints" { Eval-Checkpoints }
    "status" { Status-Run }
    "status-sweep" { Status-Sweep }
    "status-continuation" { Status-Continuation }
    "status-candidate-eval" { Status-CandidateEval }
    "status-reward-dense" { Status-RewardDense }
    "status-r7-large-eval" { Status-R7LargeEval }
    "status-r12-best-large-eval" { Status-R12BestLargeEval }
    "status-reward-r9" { Status-RewardR9 }
    "status-reward-r10" { Status-RewardR10 }
    "status-k8-pilot" { Status-K8Pilot }
    "resume-k8-pilot" { Resume-K8Pilot }
    "stop-reward-r10" { Stop-RewardR10 }
    "stop-k8-pilot" { Stop-K8Pilot }
    "fetch" { Fetch-Run }
    "fetch-sweep" { Fetch-Run }
    "fetch-continuation" { Fetch-Run }
    "fetch-candidate-eval" { Fetch-Run }
    "fetch-reward-dense" { Fetch-RewardDense }
    "fetch-r7-large-eval" { Fetch-Run }
    "fetch-r12-best-large-eval" { Fetch-Run }
    "fetch-reward-r9" { Fetch-RewardDense }
    "fetch-reward-r10" { Fetch-RewardDense }
    "fetch-k8-pilot" { Fetch-RewardDense }
    "sync-storage" { Sync-Storage }
    "restore-cache" { Restore-Cache }
    "start-tpu" { Start-Tpu }
    "stop-tpu" { Stop-Tpu }
    "delete-tpu" { Delete-Tpu }
}
