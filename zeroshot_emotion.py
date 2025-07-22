import pandas as pd
import multiprocessing as mp
from transformers import pipeline
from tqdm import tqdm
import os

# ✅ Step 1: 데이터 불러오기
CSV_PATH = "comment_only.csv"  # 여기에 파일명 넣기
COMMENT_COLUMN = "comment"  # 댓글이 담긴 컬럼명 (예: "내용", "text")

CHUNK = 10000  # 한 번에 처리할 데이터 개수

df = pd.read_csv(CSV_PATH)
texts = df[COMMENT_COLUMN].dropna().tolist()

# ✅ Step 2: 감정 레이블 정의
labels = ["긍정", "부정", "분노", "불안", "기대", "중립"]

# ✅ Step 3: 워커 함수 (프로세스마다 실행됨)
def analyze_sentiment(text):
    try:
        local_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=0,  # GPU 사용
            use_fast=False
        )
        result = local_classifier(text, labels)
        return {
            'text': text,
            'top_emotion': result['labels'][0],
            **{label: result['scores'][i] for i, label in enumerate(result['labels'])}
        }
    except Exception as e:
        return {
            'text': text,
            'top_emotion': 'error',
            **{label: None for label in labels}
        }

# ✅ Step 4: 병렬 실행 및 청크별 저장
if __name__ == "__main__":
    print("🧠 프로세스 수:", mp.cpu_count())
    total = len(texts)
    num_chunks = (total + CHUNK - 1) // CHUNK

    for i in range(num_chunks):
        chunk_texts = texts[i*CHUNK:(i+1)*CHUNK]
        print(f"▶️ {i+1}/{num_chunks}번째 청크 처리 중... ({len(chunk_texts)}개)")
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            results = list(tqdm(pool.imap(analyze_sentiment, chunk_texts), total=len(chunk_texts)))
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"감정분석_결과_{i+1:02d}.csv", index=False)
        print(f"✅ 감정분석_결과_{i+1:02d}.csv 저장 완료")

    print("🎉 전체 감정분석 완료 및 파일 분할 저장됨!")