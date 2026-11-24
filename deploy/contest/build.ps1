# 构建比赛 Harness 镜像并导出
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
        "C:\Program Files\Docker\Docker\resources\bin"
    )
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c "docker.exe")) {
            $env:PATH = "$c;" + $env:PATH
            break
        }
    }
}

docker build -f deploy/contest/Dockerfile -t zhixun-contest:1.0.4 .

New-Item -ItemType Directory -Force -Path dist | Out-Null
docker save zhixun-contest:1.0.4 -o dist/zhixun-contest-1.0.4.tar

$size = [math]::Round((Get-Item dist/zhixun-contest-1.0.4.tar).Length / 1MB, 1)
Write-Host "==> done: dist/zhixun-contest-1.0.4.tar ($size MB)" -ForegroundColor Green
