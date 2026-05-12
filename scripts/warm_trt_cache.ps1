param(
    [Parameter(Mandatory=$true)][string]$WeightsFile,
    [int]$TimeoutSec = 900
)

$ceresExe = "C:\Users\Navn\source\repos\Ceres\artifacts\release\net10.0\Ceres.exe"

Write-Host ("[warm] Starting Ceres for: {0}" -f $WeightsFile)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $ceresExe
$psi.Arguments = ""
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.WorkingDirectory = [System.IO.Path]::GetDirectoryName($ceresExe)

$proc = [System.Diagnostics.Process]::Start($psi)

$stdoutBuf = New-Object System.Text.StringBuilder
$readyOkSeen = $false
$onOut = {
    if ($EventArgs.Data) {
        $line = $EventArgs.Data
        Write-Host ("[ceres-out] {0}" -f $line)
        $script:stdoutBuf.AppendLine($line) | Out-Null
        if ($line -match "^readyok") { $script:readyOkSeen = $true }
    }
}
Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $onOut | Out-Null
Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived  -Action {
    if ($EventArgs.Data) { Write-Host ("[ceres-err] {0}" -f $EventArgs.Data) }
} | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

$si = $proc.StandardInput
$si.WriteLine("uci")
Start-Sleep -Milliseconds 500
$si.WriteLine("setoption name WeightsFile value $WeightsFile")
$si.WriteLine("setoption name Device value GPU:0#TensorRT16")
$si.WriteLine("isready")

$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $readyOkSeen -and $sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Milliseconds 250
}

if ($readyOkSeen) {
    Write-Host ("[warm] readyok received in {0:N1}s for {1}" -f $sw.Elapsed.TotalSeconds, $WeightsFile)
} else {
    Write-Host ("[warm] TIMEOUT after {0}s waiting for readyok" -f $TimeoutSec)
}

$si.WriteLine("quit")
$proc.WaitForExit(5000) | Out-Null
if (-not $proc.HasExited) { $proc.Kill() }
Get-EventSubscriber | Unregister-Event
Write-Host "[warm] done"
