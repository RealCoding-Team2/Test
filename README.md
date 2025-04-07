# 📌MediaPipe 테스트
이 웹 애플리케이션은 머신러닝 기반 이미지 및 동영상 딥페이크 탐지 서비스입니다. 사용자가 업로드한 미디어 파일의 딥페이크 가능성을 분석하고 결과를 제공합니다.
## ✨ 주요 기능

이미지 딥페이크 탐지
동영상 딥페이크 탐지
실시간 분석 결과 제공
딥페이크 신뢰도 점수 표시

## 🛠 기술 스택

- Backend: Flask (Python)
- 컴퓨터 비전: OpenCV, MediaPipe
- 머신러닝 분석: NumPy
- 얼굴 탐지: MediaPipe Face Detection

## 🖥 화면 구성

![page-1](https://github.com/user-attachments/assets/69c5ea47-a38b-4a31-88df-d9e4ba0905c0)
1. 메인 페이지 (업로드 인터페이스)
- 파일 업로드 기능
- 지원 파일 형식: JPG, PNG, MP4, AVI, MOV

![page-2](https://github.com/user-attachments/assets/72b4aef7-8b67-4b3f-b206-0119d09cc519)
2. 결과 상세 정보 화면
  - 이미지/영상 특성 분석
  - 노이즈, 경계, 색상, 텍스처 분석 결과

## 🔍 분석 방법

MediaPipe를 활용한 얼굴 탐지
이미지/영상의 다양한 특성 분석

노이즈 레벨
경계선 일관성
색상 분포
텍스처 특성


머신러닝 기반 딥페이크 스코어 계산
## 📦 설치 및 실행

### pip 업그레이드
python -m pip install --upgrade pip

#### 의존성 패키지 설치
pip install flask
pip install opencv-python
pip install mediapipe
pip install numpy
- python 확인.py 실행

## 📋 사용 방법

1. 웹 브라우저에서 애플리케이션 접속
2. 분석할 이미지 또는 동영상 파일 업로드
3. 분석 결과 확인

## ⚠️ 주의사항

본 서비스는 딥페이크 탐지를 위한 실험적 도구입니다.
100% 정확한 탐지는 어렵습니다.
다양한 요인에 따라 분석 결과가 달라질 수 있습니다.
