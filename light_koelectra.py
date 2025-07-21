from transformers import pipeline
from tqdm import tqdm
import pandas as pd

# 댓글 로딩
df = pd.read_csv("comment_only.csv")
texts = df["comment"].dropna().tolist()

# ✅ KoELECTRA 기반 경량 Zero-shot 모델 사용monologg/koelectra-base-v3-nli

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    framework="pt",  # ← 이것이 핵심! 강제로 PyTorch 사용

    device=-1  # CPU 사용 (GPU 가능 시 0)
)

# 감정 레이블
labels = ["긍정", "부정", "분노", "불안", "기대", "중립"]

# 분석 수행
results = []
for text in tqdm(texts[:1000]):  # ← 처음엔 1000개만 테스트!
    try:
        out = classifier(text, labels)
        results.append({
            'text': text,
            'top_emotion': out['labels'][0],
            **{label: score for label, score in zip(out['labels'], out['scores'])}
        })
    except:
        results.append({
            'text': text,
            'top_emotion': 'error',
            **{label: None for label in labels}
        })

# 결과 저장
pd.DataFrame(results).to_csv("koelectra_감정분석_샘플.csv", index=False)