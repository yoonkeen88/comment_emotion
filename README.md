# 한국어 댓글 감정 분석 프로젝트

이 프로젝트는 한국어 온라인 댓글 데이터를 분석하여 긍정, 비판, 분노, 불안, 기대, 중립의 6가지 감정을 분류하는 모델을 구현하고 실행하는 것을 목표로 합니다.

## 주요 기능

- **욕설 및 비속어 처리**: `badword.py` 와 `Emotion_dictionary.py`를 사용하여 댓글에 포함된 욕설과 비속어를 탐지하고 정제합니다.
- **감정 분석**: `joeddav/xlm-roberta-large-xnli` 모델을 활용하여 댓글의 감정을 다각적으로 분석합니다.
- **대용량 데이터 처리**: 대규모 데이터셋을 효율적으로 처리하기 위해 데이터를 청크(chunk) 단위로 나누어 분석하고, 중간 결과를 저장하여 작업 중단 시에도 복구할 수 있습니다.

## 기술 스택 및 모델

- **프로그래밍 언어**: Python 3.9.13
- **핵심 라이브러리**: PyTorch, Hugging Face Transformers, Pandas
- **감정 분석 모델**: `joeddav/xlm-roberta-large-xnli` (Zero-shot Classification)

## 설치 및 실행 방법

1.  **저장소 복제**

    ```bash
    git clone https://github.com/your-username/emotion_analysis.git
    cd emotion_analysis
    ```

2.  **가상 환경 생성 및 활성화**

    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **의존성 라이브러리 설치**

    프로젝트에 필요한 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어로 한 번에 설치할 수 있습니다.

    ```bash
    pip install -r requirements.txt
    ```

4.  **감정 분석 실행**

    `감정분석1.py` 스크립트를 사용하여 감정 분석을 실행합니다. 스크립트는 다음과 같은 인자(argument)를 받습니다.

    -   `--input`: 분석할 댓글 데이터가 포함된 CSV 파일 경로 (필수)
    -   `--output_dir`: 분석 결과를 저장할 폴더 (기본값: `result_parts`)
    -   `--chunk_size`: 한 번에 처리할 데이터의 양 (기본값: 1000)
    -   `--start_chunk`: 분석을 시작할 청크 번호 (기본값: 1). 작업 중단 시 복구에 사용됩니다.

    **실행 예시:**

    ```bash
    python 감정분석1.py --input data/comments.csv --output_dir results --chunk_size 2000 --start_chunk 53
    ```

    위 예시는 `data/comments.csv` 파일을 2000개씩 나누어 분석하며, 53번째 청크부터 작업을 시작하여 `results` 폴더에 결과를 저장합니다.

## 프로젝트 구조

```
.emotion_analysis/
├───.gitignore
├───감정분석.py
├───감정분석1.py
├───badword.py
├───Emotion_dictionary.py
├───README.md
├───requirements.txt
├───.venv/
└───...
```

-   `감정분석1.py`: 메인 감정 분석 스크립트
-   `badword.py`, `Emotion_dictionary.py`: 텍스트 전처리를 위한 욕설/비속어 사전 및 관련 로직
-   `requirements.txt`: 프로젝트 의존성 라이브러리 목록
-   `result_parts/` (생성됨): 분석 결과가 청크별 CSV 파일로 저장되는 폴더