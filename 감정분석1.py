import torch
import subprocess

def print_gpu_memory():
    if torch.cuda.is_available():
        print("🖥️ GPU 상태 확인:")
        try:
            # torch 기반 체크
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"  - CUDA 메모리 사용량: {allocated:.2f} MB (할당됨), {reserved:.2f} MB (예약됨)")

            # nvidia-smi로도 체크 (선택사항)
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("  - NVIDIA-SMI 상태 ↓")
            print(result.stdout.split("Processes")[1])  # 출력 일부만 표시
        except Exception as e:
            print(f"⚠️ GPU 정보 확인 중 오류: {e}")
    else:
        print("❌ CUDA 사용 불가: GPU를 찾을 수 없음.")
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import os
import argparse
import torch

LABELS = ["긍정", "비판", "분노", "불안", "기대", "중립"]
COMMENT_COLUMN = "comment"
DATE_COLUMN = "date"

# ✅ 감정 분석기 초기화
def init_model():
    device = 0 if torch.cuda.is_available() else -1
    print(f"✅ 감정분석 모델 로딩 중 (device={device})")
    return pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        framework="pt",
        device=device
    )

# ✅ 개별 문장 처리
def analyze_sentiment(text, classifier):
    try:
        output = classifier(text, LABELS)
        return {
            "top_emotion": output["labels"][0],
            **{label: output["scores"][i] for i, label in enumerate(output["labels"])}
        }
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
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
    classifier = init_model()

    start_chunk_index = args.start_chunk - 1

    for i, chunk in enumerate(chunks):
        if i < start_chunk_index:
            continue

        save_path = os.path.join(args.output_dir, f"감정분석_결과_part{i+1}.csv")
        if os.path.exists(save_path):
            print(f"⏭️ 파일이 이미 존재하므로 청크 {i+1}를 건너<binary data, 2 bytes>니다: {save_path}")
            continue

        print(f"🔄 청크 {i+1}/{len(chunks)} 분석 중...")
        print_gpu_memory()  # ✅ GPU 상태 출력
        results = [analyze_sentiment(text, classifier) for text in tqdm(chunk)]

        result_df = pd.DataFrame(results)
        original_meta = df.iloc[i * args.chunk_size : i * args.chunk_size + len(chunk)].reset_index(drop=True)
        merged_df = pd.concat([original_meta, result_df], axis=1)

        merged_df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

    print("🎉 전체 분석 완료!")

# ✅ argparse 인자 처리
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="분석할 CSV 파일 경로")
    parser.add_argument("--output_dir", default="result_parts", help="결과 저장 폴더")
    parser.add_argument("--chunk_size", type=int, default=1000, help="청크 크기")
    parser.add_argument("--start_chunk", type=int, default=1, help="시작할 청크 번호")
    args = parser.parse_args()

    main(args)
