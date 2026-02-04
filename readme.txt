1) 요구사항
- Windows + PowerShell
- Python 3.11.9 
- ffmpeg 

2) 가상환경 생성/설치
- 프로젝트 최상위 폴더에서 실행 (PowerShell 기준)
  python -m venv venv 
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt

3) 실행 방법
- 가상환경 활성화 후 아래 순서대로 실행
  .\venv\Scripts\Activate.ps1
  cd romance_detection
  python webcam.py
