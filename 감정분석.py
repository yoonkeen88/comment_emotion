import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from transformers import pipeline
import os
import argparse

LABELS = ["긍정", "비판", "분노", "불안", "기대", "중립"]
COMMENT_COLUMN = "comment"
DATE_COLUMN = "date"

# ✅ 감정 분석기 초기화
def init_model():
    global classifier
    classifier = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        framework="pt",
        device=-1
    )

# ✅ 개별 문장 처리
def analyze_sentiment(text):
    try:
        output = classifier(text, LABELS)
        return {
            "top_emotion": output["labels"][0],
            **{label: output["scores"][i] for i, label in enumerate(output["labels"])}
        }
    except:
        return {
            "top_emotion": "error",
            **{label: None for label in LABELS}
        }

# ✅ 메인 실행
def main(args):
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[COMMENT_COLUMN]).reset_index(drop=True)
    comments = df[COMMENT_COLUMN].tolist()
    total = len(comments)
    chunks = [comments[i:i + args.chunk_size] for i in range(0, total, args.chunk_size)]

    os.makedirs(args.output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        print(f"🔄 청크 {i+1}/{len(chunks)} 분석 중...")
        with mp.Pool(processes=args.num_proc, initializer=init_model) as pool:
            results = list(tqdm(pool.imap(analyze_sentiment, chunk), total=len(chunk)))

        result_df = pd.DataFrame(results)
        original_meta = df.iloc[i * args.chunk_size : i * args.chunk_size + len(chunk)].reset_index(drop=True)
        merged_df = pd.concat([original_meta, result_df], axis=1)

        save_path = os.path.join(args.output_dir, f"감정분석_결과_part{i+1}.csv")
        merged_df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

    print("🎉 전체 분석 완료!")

# ✅ argparse 인자 처리
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="분석할 CSV 파일 경로")
    parser.add_argument("--output_dir", default="result_parts", help="결과 저장 폴더")
    parser.add_argument("--chunk_size", type=int, default=1000, help="청크 크기")
    # ⬇️ argparse에 추가
    parser.add_argument("--num_proc", type=int, default=2, help="사용할 병렬 프로세스 수")
    args = parser.parse_args()

    main(args)