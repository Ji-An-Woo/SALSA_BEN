# SALSA-repro 완전 자동화 실행 PowerShell 스크립트
# ====================================================

Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host "🚀 SALSA-repro 완전 자동화 실행 v2.0" -ForegroundColor Yellow
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host ""

# 현재 위치 확인
if (!(Test-Path "src\")) {
    Write-Host "❌ 오류: src 폴더를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "   salsa-repro-salsa-connected 폴더에서 실행하세요." -ForegroundColor Red
    Write-Host ""
    Read-Host "엔터 키를 누르면 종료합니다"
    exit 1
}

Write-Host "🎯 완전 자동화 스크립트를 실행합니다..." -ForegroundColor Green
Write-Host "   - GPU/CUDA 설정" -ForegroundColor White
Write-Host "   - 패키지 설치" -ForegroundColor White
Write-Host "   - 데이터 생성" -ForegroundColor White
Write-Host "   - SALSA 훈련" -ForegroundColor White
Write-Host "   - 결과 평가" -ForegroundColor White
Write-Host ""
Write-Host "⏰ 예상 소요 시간: 5-30분 (GPU 성능에 따라)" -ForegroundColor Yellow
Write-Host ""

Read-Host "계속하려면 엔터 키를 누르세요"

Write-Host "🚀 Python 스크립트 실행 중..." -ForegroundColor Cyan
Write-Host ""

# Python 스크립트 실행
try {
    py run_complete.py
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "✅ 실행 완료!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 결과 확인:" -ForegroundColor Yellow
        Write-Host "   - results\salsa_runs\salsa_summary.csv" -ForegroundColor White
        Write-Host "   - results\salsa_runs\salsa_summary.json" -ForegroundColor White
    } else {
        Write-Host "⚠️ 일부 문제가 발생했지만 실행이 완료되었습니다." -ForegroundColor Yellow
        Write-Host "   results\ 폴더에서 결과를 확인하세요." -ForegroundColor White
    }
} catch {
    Write-Host "❌ 실행 중 오류가 발생했습니다: $_" -ForegroundColor Red
}

Write-Host ""
Read-Host "아무 키나 누르면 종료합니다"