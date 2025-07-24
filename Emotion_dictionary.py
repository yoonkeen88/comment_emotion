import pandas as pd

# 감정별 대표 단어 사전 구성
emotion_dict = {
    "긍정": ["좋다", "기쁘다", "만족", "행복", "사랑", "감사", "즐겁다", "잘했다", "칭찬", "훌륭하다"],
    "기대": ["기대", "설레다", "기다리다", "희망", "바라다", "궁금하다", "흥미", "관심", "기대감", "희망적"],
    "비판": ["문제", "부족", "개선", "지적", "비효율", "불편", "불만", "우려", "불확실", "의문"],
    "실망": ["실망", "아쉽다", "후회", "짜증", "지치다", "불쾌", "낙담", "짜증나다", "허탈", "한숨"],
    "분노": ["화나다", "열받다", "짜증", "빡치다", "분노", "역겹다", "폭발", "싫다", "짜증나", "폭력적"],
    "중립": ["그렇다", "그냥", "보통", "평범", "무난", "잘모르겠다", "아무생각없다", "중립", "평범하다", "일반적"]
}

# 데이터를 리스트로 변환
rows = []
for emotion, keywords in emotion_dict.items():
    for word in keywords:
        rows.append({"emotion": emotion, "word": word})

# 데이터프레임 생성
df = pd.DataFrame(rows)

# 저장
output_path = "/mnt/data/emotion_dictionary.csv"
df.to_csv(output_path, index=False)