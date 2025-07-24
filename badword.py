import pandas as pd
import re
from multiprocessing import Pool, cpu_count
import os
import time

# 감정별 대표 단어 사전 (emotion_dictionary.py 기반)
emotion_dict = {
    "긍정": ["좋다", "기쁘다", "만족", "행복", "사랑", "감사", "즐겁다", "잘했다", "칭찬", "훌륭하다"],
    "기대": ["기대", "설레다", "기다리다", "희망", "바라다", "궁금하다", "흥미", "관심", "기대감", "희망적"],
    "비판": ["문제", "부족", "개선", "지적", "비효율", "불편", "불만", "우려", "불확실", "의문"],
    "실망": ["실망", "아쉽다", "후회", "지치다", "불쾌", "낙담", "허탈", "한숨"],
    "분노": ["화나다", "열받다", "짜증", "빡치다", "분노", "역겹다", "폭발", "싫다", "짜증나", "폭력적"],
    "중립": ["그렇다", "그냥", "보통", "평범", "무난", "잘모르겠다", "아무생각없다", "중립", "평범하다", "일반적"]
}

# 욕설, 비속어, 신조어, 줄임말 정규화 사전
normalization_dict = {
    # ======================================================
    # 1. 심한 욕설 및 비하 표현
    # ======================================================
    r'개돼지|개돼쥐': '비하 발언',
    r'(ㅅ|ㅆ)(ㅂ|발|팔|벌|바|파|ㅍ|ㅃ)': '욕설',
    r'(ㅈ|좆|졷|조ㅈ)(같|가|까)': '매우 별로인',
    r'(ㅈ|즂|ㅈㅗㅈ|ㅈㅗㅅ)ㄴ': '매우',
    r'(ㅁ|미)(ㅊ|친|췬|친)': '비정상적인',
    r'(ㅂ|병)(ㅅ|신|쉰)': '바보',
    r'(ㅈ|지)(ㄹ|랄|럴)': '헛소리',
    r'(ㄷ|등)(ㅅ|신)': '바보',
    r'(ㄱ|개)(ㅅ|새|섀)(ㄲ|끼)': '나쁜놈',
    r'새끼': '비하',
    r'꺼져': '나가',
    r'존나|졸라': '매우',
    r'아가리': '입',
    r'닥쳐': '조용히 해',

    # ======================================================
    # 2. 정치/사회 관련 은어 및 비하 표현
    # ======================================================
    r'기레기': '기자 비하',
    r'문재앙|문죄인': '문재인 대통령 비하',
    r'박그네|닭그네': '박근혜 대통령 비하',
    r'이명박|쥐박이': '이명박 대통령 비하',
    r'틀딱충|틀딱': '노인 비하',
    r'맘충': '어머니 비하',
    r'한녀충|한남충': '성별 비하',
    r'좌빨|좌좀': '좌파 비하',
    r'수구꼴통': '우파 비하',
    r'대깨문': '문재인 대통령 지지자 비하',

    # ======================================================
    # 3. 일반 은어, 줄임말, 초성
    # ======================================================
    # --- 긍정 ---
    r'존맛|존맛탱|JMT': '매우 맛있는',
    r'꿀잼': '매우 재미있는',
    r'개이득': '큰 이익',
    r'핵잼': '매우 재미있는',
    # --- 부정 ---
    r'노잼': '재미없는',
    r'극혐': '매우 혐오스러운',
    r'헬조선|헬.+': '매우 힘든 상황',
    # --- 초성 ---
    r'ㅇㅈ': '인정',
    r'ㄹㅇ': '정말',
    r'ㄱㅅ': '감사',
    r'ㅅㅌㅊ': '상위권',
    r'ㅍㅌㅊ': '평균',
    r'ㅎㅌㅊ': '하위권',
    r'ㄷㄷ': '놀람',
    r'ㅂㄷㅂㄷ': '부들부들',
    r'ㅇㅇ': '응',
    r'ㄴㄴ': '아니',
    r'ㅉㅉ': '짜증',
    # --- 일반 줄임말 ---
    r'걍': '그냥',
    r'넘': '너무',

    # ======================================================
    # 4. 감정 표현 이모티콘 및 기타
    # ======================================================
    r'\bㅋ\b': '비웃음',
    r'[ㅋ]{2,}': '웃음',
    r'[ㅎ]{2,}': '웃음',
    r'[ㅠㅜ]{2,}': '슬픔',
    r'[!?]{2,}': '강조',
    r'[ㅡ]{2,}|--': '불만',
    r'\^\^': '미소',
    r'ㄷㄷ': '놀람',
}

def preprocess_text(text):
    """
    댓글 텍스트에 대한 전체 전처리 파이프라인입니다.
    1. 욕설, 은어, 줄임말을 정규화합니다.
    2. '개' 접두사 및 '~충', '~놈' 등 접미사를 처리합니다.
    3. 반복되는 자음, 모음, 특수문자를 처리합니다.
    4. 불필요한 특수문자를 제거합니다.
    5. 과도한 공백을 제거합니다.
    6. 감성 사전을 기반으로 감정 라벨을 추가합니다.
    """
    text = str(text).lower()

    # 1. 정규화 사전 적용
    for pattern, replacement in normalization_dict.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            continue

    # 2. 접두사 및 접미사 처리
    text = re.sub(r'\b개(?!새)', '매우 ', text)
    text = re.sub(r'[가-힣]+충\b', ' 비하', text)
    text = re.sub(r'[가-힣]+(놈|년)\b', ' 사람', text)

    # 3. 반복 문자 처리
    text = re.sub(r'([ㅋㅎ])\1{1,}', ' 웃음 ', text)
    text = re.sub(r'([ㅠㅜ])\1{1,}', ' 슬픔 ', text)
    text = re.sub(r'([ㅏ-ㅣ])\1{2,}', r'\1', text)
    text = re.sub(r'([!?.,])\1{1,}', r'\1', text)

    # 4. 불필요한 특수문자 제거
    text = re.sub(r'[^\w\s.?!가-힣]', ' ', text)

    # 5. 과도한 공백 제거
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # 6. 감성 사전 기반 라벨링
    found_emotions = []
    for emotion, keywords in emotion_dict.items():
        for keyword in keywords:
            if keyword in text:
                found_emotions.append(emotion)
                break  # 한 감정 카테고리에서 단어가 발견되면 다음 카테고리로 넘어감

    if found_emotions:
        # 중복을 제거하고 알파벳 순으로 정렬하여 일관성 유지
        unique_emotions = sorted(list(set(found_emotions)))
        emotion_label = f" (감정: {', '.join(unique_emotions)})"
        text += emotion_label

    return text

def process_row(row):
    comment = str(row.get('comment', ''))
    processed_comment = preprocess_text(comment)
    if processed_comment:
        return {'date': row.get('date'), 'comment': processed_comment}
    return None

def process_file_parallel(input_path):
    try:
        try:
            df = pd.read_csv(os.path.join('split_data', input_path), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(os.path.join('split_data', input_path), encoding='cp949')
        
        rows = df.to_dict(orient='records')
        with Pool(cpu_count()) as pool:
            results = pool.map(process_row, rows)
        
        filtered_results = [r for r in results if r is not None]
        return pd.DataFrame(filtered_results)
    except FileNotFoundError:
        print(f"경고: {input_path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

def main():
    print(f"전처리 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    input_dir = 'split_data'
    output_dir = 'processed_comments'
    os.makedirs(output_dir, exist_ok=True)

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    all_processed_dfs = []
    for file_name in input_files:
        print(f"▶ 처리 시작: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        processed_df = process_file_parallel(file_name) # file_name만 전달

        if not processed_df.empty:
            print(f"✅ {file_name} 처리 완료, 건수: {len(processed_df)}")
            output_path = os.path.join(output_dir, f'processed_{file_name}')
            processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            all_processed_dfs.append(processed_df)
        else:
            print(f"⚠️ {file_name} 처리 결과가 비어있습니다.")

    if all_processed_dfs:
        final_df = pd.concat(all_processed_dfs, ignore_index=True)
        final_output_path = os.path.join(output_dir, 'final_processed_comments.csv')
        final_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
        print(f"🎉 전체 데이터 처리 및 병합 완료, 총 건수: {len(final_df)}")
    else:
        print("처리할 데이터가 없어 최종 파일을 생성하지 않았습니다.")

    print(f"전처리 종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()