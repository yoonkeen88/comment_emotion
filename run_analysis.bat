@echo off
echo [INFO] 감정 분석 2개 세션을 실행합니다...

REM 첫 번째 분석 - part1
start "감정분석_1" powershell -NoExit -Command "python 감정분석.py --input split_data/comments_part1.csv --output_dir result_win1 --chunk_size 2000 --num_proc 2"

REM 두 번째 분석 - part2
start "감정분석_2" powershell -NoExit -Command "python 감정분석.py —input split_data/comments_part2.csv —output_dir result_win2 —chunk_size 2000 —num_proc 2"

echo [INFO] 모든 분석 명령이 실행되었습니다. 터미널에서 진행 상황을 확인하세요.
pause