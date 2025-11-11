@echo off
echo ========================================
echo    SALSA-repro 완전 자동화 실행
echo ========================================
echo.

REM 현재 디렉토리 확인
if not exist "src\" (
    echo ❌ 오류: src 폴더가 없습니다. salsa-repro-salsa-connected 폴더에서 실행하세요.
    pause
    exit /b 1
)

echo 🚀 Python 스크립트 실행 중...
echo.

py run_all.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 실행 완료! 결과를 확인하세요.
    echo 📊 결과 위치: results\salsa_runs\salsa_summary.csv
) else (
    echo.
    echo ❌ 실행 중 오류가 발생했습니다.
)

echo.
echo 아무 키나 누르면 종료합니다...
pause >nul