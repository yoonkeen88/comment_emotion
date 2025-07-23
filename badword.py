import pandas as pd
import re
from multiprocessing import Pool, cpu_count
import os

# 욕설 탐지용 정규식 패턴
profanity_patterns = [
    r'ㅈ+같[다은]',
    r'ㅅ+ㅂ',
    r'ㅂ+ㅅ',
    r'ㄱ+ㅅㄲ',
    r'좆',
    r'씨+발',
    r'미+친',
    r'ㅈ+ㄴ',
]

# 욕설 정규화 룰
def normalize_profanity(text):
    patterns = {
        r'ㅈ+같[다은]': '짜증난다',
        r'ㅅ+ㅂ': '진짜 열받는다',
        r'ㅂ+ㅅ': '답답한 사람',
        r'ㄱ+ㅅㄲ': '불쾌한 사람',
        r'좆': '정말 싫은',
        r'씨+발': '정말 화가 난',
        r'미+친': '말도 안 되는',
        r'ㅈ+ㄴ': '진짜',
        r'졸라': '매우',
        r'개+같[다은]': '매우 짜증나는',
        r'ㅉㅉ': '짜증나',
        r'ㅋ' : '웃음',
        r'ㅎ' : '웃음',
    }
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    return text

# 한국형 이모지 제거용 정규식 패턴
korean_emojis_patterns = [
    r'[ㅋㅎ]{2,}',        # ㅋㅋ, ㅎㅎ
    r'[ㅜㅠ]{1,}',        # ㅠㅠ, ㅜㅜ
    r'[-~_^]{2,}',        # ^^, ^~^, ~~ 등
    r'ㅡ\.ㅡ', r'ㅇ_ㅇ',   # 얼굴형 이모지
    r'ㅡ',
    r'~',
    r'>_<', r'^_^', r'T_T',
    r'헐', r'읭', r'웅', r'엥', r'아놔'
]

def remove_korean_emojis(text):
    for pattern in korean_emojis_patterns:
        text = re.sub(pattern, '', text)
    return text

def contains_profanity(text):
    for pat in profanity_patterns:
        if re.search(pat, text):
            return True
    return False

def process_row(row):
    text = str(row['comment'])
    if contains_profanity(text):
        normalized = normalize_profanity(text)
        normalized = remove_korean_emojis(normalized)
        if contains_profanity(normalized):
            return None
        else:
            return {'date': row['date'], 'comment': normalized}
    else:
        clean = remove_korean_emojis(text)
        return {'date': row['date'], 'comment': clean}

def process_file_parallel(input_path):
    df = pd.read_csv(f'split_data/{input_path}')
    rows = df.to_dict(orient='records')

    with Pool(cpu_count()) as pool:
        results = pool.map(process_row, rows)

    filtered = [r for r in results if r is not None]
    return pd.DataFrame(filtered)

import time
def main():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    input_files = [f'comments_part{i}.csv' for i in range(1, 4)]
    os.makedirs('processed_comments', exist_ok=True)

    all_dfs = []
    for file in input_files:
        print(f"▶ 처리 시작: {file}")
        processed_df = process_file_parallel(file)
        print(f"✅ {file} 처리 완료, 건수: {len(processed_df)}")
        all_dfs.append(processed_df)

        processed_df.to_csv(f'processed_comments/processed_{file}', index=False)

    final_df = pd.concat(all_dfs).reset_index(drop=True)
    final_df.to_csv('processed_comments/final_processed_comments.csv', index=False)
    print(f"🎉 전체 데이터 처리 완료, 총 건수: {len(final_df)}")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

if __name__ == "__main__":
    main()
