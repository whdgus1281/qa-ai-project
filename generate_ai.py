# 파이썬 → AI 학습 → 버그 예측 → HTML 자동 생성

# =============================================
# 라이브러리 불러오기
# =============================================

# sklearn = 머신러닝 도구 모음
# DecisionTreeClassifier = 결정트리 모델
# "스무고개처럼 질문해서 분류하는 AI"
from sklearn.tree import DecisionTreeClassifier

# train_test_split = 데이터를 학습용/시험용으로 나눠주는 함수
# AI도 공부하고 시험을 봐야 해요!
from sklearn.model_selection import train_test_split

# accuracy_score = AI가 몇 % 맞췄는지 계산해주는 함수
from sklearn.metrics import accuracy_score

# pandas = 데이터를 엑셀 표처럼 다루는 도구
import pandas as pd

# =============================================
# ① 학습 데이터 준비
# =============================================

# X = AI에게 줄 입력값 (문제)
# 각 항목: [응답시간(초), 에러코드(1=있음/0=없음), 하루발생횟수]
# [[...]] 처럼 2차원 리스트로 만들어야 해요 (sklearn 규칙!)
X_train_data = [
    [0.5, 0, 1],   # 빠르고, 에러없고, 1번 발생
    [0.8, 0, 2],   # 빠르고, 에러없고, 2번 발생
    [1.2, 0, 1],   # 보통, 에러없고, 1번 발생
    [1.5, 1, 3],   # 보통, 에러있고, 3번 발생
    [2.1, 1, 5],   # 느리고, 에러있고, 5번 발생
    [3.2, 1, 8],   # 매우느리고, 에러있고, 8번 발생
    [4.5, 1, 10],  # 매우느리고, 에러있고, 10번 발생
    [5.0, 1, 15],  # 최악, 에러있고, 15번 발생
    [0.3, 0, 1],   # 매우빠르고, 에러없고, 1번 발생
    [2.5, 1, 6],   # 느리고, 에러있고, 6번 발생
    [1.8, 0, 2],   # 보통, 에러없고, 2번 발생
    [3.8, 1, 9],   # 매우느리고, 에러있고, 9번 발생
]

# y = 정답 레이블 (답)
# 0 = Minor (별로 안 심각)
# 1 = Critical (매우 심각)
# X_train_data와 순서가 정확히 같아야 해요!
y_train_data = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
#               ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#              0.5 0.8 1.2 1.5 2.1 3.2 4.5 5.0 0.3 2.5 1.8 3.8

# =============================================
# ② AI 학습시키기
# =============================================

# train_test_split = 데이터를 학습용/시험용으로 자동으로 나눠줘요
# test_size=0.2 → 20%는 시험용, 80%는 학습용
# random_state=42 → 매번 똑같은 방식으로 나눠요 (재현 가능하게)
# X_tr = 학습용 입력, X_te = 시험용 입력
# y_tr = 학습용 정답, y_te = 시험용 정답
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_data, y_train_data,
    test_size=0.2,
    random_state=42
)

# DecisionTreeClassifier() = 결정트리 AI 모델 생성
# 아직 아무것도 모르는 빈 AI예요
model = DecisionTreeClassifier()

# model.fit() = AI 학습!
# "이 입력(X_tr)의 정답은 이거야(y_tr)!" 라고 가르치는 것
# 사람으로 치면 교과서로 공부하는 단계
model.fit(X_tr, y_tr)

# model.predict() = 시험용 데이터로 예측
# AI가 배운 것을 바탕으로 답을 맞춰봐요
# accuracy_score() = 예측한 것(y_pred)과 정답(y_te) 비교
# * 100 → 퍼센트로 변환
accuracy = accuracy_score(y_te, model.predict(X_te)) * 100

# =============================================
# ③ 새 버그 데이터 — AI가 판단할 버그들
# =============================================

# 실제로 AI한테 분류시킬 새 버그들이에요
# 아직 심각도를 모르는 상태 → AI가 판단해줄 거예요
new_bugs = [
    {"id": "BUG-001", "name": "로그인 500 에러",  "time": 4.2, "error": 1, "count": 12},
    {"id": "BUG-002", "name": "메뉴 아이콘 깨짐", "time": 0.5, "error": 0, "count": 2},
    {"id": "BUG-003", "name": "결제 타임아웃",    "time": 5.1, "error": 1, "count": 20},
    {"id": "BUG-004", "name": "오타 수정",         "time": 0.3, "error": 0, "count": 1},
    {"id": "BUG-005", "name": "검색 느림",         "time": 3.1, "error": 1, "count": 7},
    {"id": "BUG-006", "name": "버튼 색상 오류",    "time": 0.4, "error": 0, "count": 3},
]

# =============================================
# ④ AI로 심각도 예측
# =============================================

# 딕셔너리 리스트 → Pandas DataFrame(표)으로 변환
df = pd.DataFrame(new_bugs)

# df[['time', 'error', 'count']] → time, error, count 열만 꺼내기
# .values.tolist() → numpy 배열을 파이썬 리스트로 변환
# AI 입력값으로 쓸 수 있는 형태로 만드는 거예요
X_new = df[['time', 'error', 'count']].values.tolist()

# model.predict() = 드디어 AI 예측!
# X_new의 각 버그를 보고 0(Minor) 또는 1(Critical) 반환
predictions = model.predict(X_new)
# predictions = [1, 0, 1, 0, 1, 0] 이런 식으로 나와요

# predictions 결과를 보기 좋게 텍스트로 변환해서 df에 추가
# p == 1 이면 'Critical', 아니면 'Minor'
# [... for p in predictions] = 리스트 컴프리헨션 (반복문 짧게 쓰기)
df['severity'] = ['Critical' if p == 1 else 'Minor' for p in predictions]

# =============================================
# ⑤ 통계 계산
# =============================================

# 전체 버그 수
total = len(df)

# severity 열에서 'Critical'인 것만 필터링 → 개수 세기
critical = len(df[df['severity'] == 'Critical'])

# severity 열에서 'Minor'인 것만 필터링 → 개수 세기
minor = len(df[df['severity'] == 'Minor'])

# =============================================
# ⑥ HTML 표 행(tr) 자동 생성
# =============================================

# 나중에 HTML 표 안에 넣을 내용
# 지금은 빈 문자열로 시작
rows = ""

# df.iterrows() = 표의 각 행을 하나씩 꺼내요
# _ = 행 번호 (안 쓸 거라 _로 무시)
# row = 그 행의 데이터 (id, name, time 등)
for _, row in df.iterrows():

    # severity에 따라 다른 뱃지 HTML 만들기
    if row['severity'] == 'Critical':
        badge = '<span class="critical">🔴 Critical</span>'
    else:
        badge = '<span class="minor">🟡 Minor</span>'

    # f""" """ = 여러 줄 f-string
    # {row['id']} = 파이썬 변수를 HTML 안에 자동으로 넣어줘요
    # row['error'] == 1 이면 '있음', 아니면 '없음' 출력
    # += = rows에 계속 이어붙이기
    rows += f"""
    <tr>
      <td class="id">{row['id']}</td>
      <td>{row['name']}</td>
      <td>{row['time']}초</td>
      <td>{'있음' if row['error'] == 1 else '없음'}</td>
      <td>{row['count']}회</td>
      <td>{badge}</td>
    </tr>"""

# =============================================
# ⑦ HTML 파일 전체 생성
# =============================================

# f""" """ 로 HTML 전체를 문자열로 만들어요
# {변수} 자리에 파이썬 값이 자동으로 들어가요
# CSS의 { } 는 {{ }} 로 써야해요
# 파이썬이 CSS { } 를 변수로 착각하지 않게!
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>QA AI 대시보드</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: sans-serif; background: #0d0d0d; color: #f0f0f0; padding: 40px; }}
    h1 {{ font-size: 22px; color: #4ade80; margin-bottom: 6px; }}
    .subtitle {{ font-size: 13px; color: #555; margin-bottom: 32px; }}
    .stats {{ display: flex; gap: 16px; margin-bottom: 32px; }}
    .stat {{ background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 20px 28px; text-align: center; flex: 1; }}
    .stat-num {{ font-size: 40px; font-weight: 700; line-height: 1; margin-bottom: 6px; }}
    .stat-label {{ font-size: 12px; color: #555; }}
    .c-green {{ color: #4ade80; }}
    .c-red   {{ color: #f87171; }}
    .c-yellow{{ color: #fbbf24; }}
    table {{ width: 100%; border-collapse: collapse; background: #111; border-radius: 12px; overflow: hidden; }}
    th {{ background: #161616; color: #555; padding: 12px 16px; text-align: left; font-size: 11px; border-bottom: 1px solid #1e1e1e; }}
    td {{ padding: 12px 16px; border-bottom: 1px solid #161616; font-size: 14px; }}
    tr:last-child td {{ border-bottom: none; }}
    .id {{ font-family: monospace; font-size: 12px; color: #555; }}
    .critical {{ background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.2); color: #f87171; padding: 4px 12px; border-radius: 100px; font-size: 12px; }}
    .minor {{ background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.2); color: #fbbf24; padding: 4px 12px; border-radius: 100px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>🤖 QA AI 대시보드</h1>
  <p class="subtitle">머신러닝이 버그 심각도를 자동으로 판단해요</p>

  <div class="stats">
    <div class="stat">
      <div class="stat-num">{total}</div>
      <div class="stat-label">전체 버그</div>
    </div>
    <div class="stat">
      <div class="stat-num c-red">{critical}</div>
      <div class="stat-label">🔴 Critical</div>
    </div>
    <div class="stat">
      <div class="stat-num c-yellow">{minor}</div>
      <div class="stat-label">🟡 Minor</div>
    </div>
    <div class="stat">
      <div class="stat-num c-green">{accuracy:.0f}%</div>
      <div class="stat-label">AI 정확도</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>버그명</th>
        <th>응답시간</th>
        <th>에러코드</th>
        <th>발생횟수</th>
        <th>AI 판단</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>"""

# =============================================
# ⑧ 파일로 저장
# =============================================

# open("파일명", "w") = 파일을 쓰기 모드로 열기
# "w" = write(쓰기 모드)
# encoding="utf-8" = 한글 깨짐 방지
# as f = 이 파일을 f라는 이름으로 부를게요
with open("qa_ai_dashboard.html", "w", encoding="utf-8") as f:
    # f.write(html) = html 문자열을 파일에 저장
    f.write(html)

# 완료 메시지!
print("✅ qa_ai_dashboard.html 생성 완료!")
print(f"   AI 정확도: {accuracy:.1f}%")
print(f"   Critical: {critical}개 | Minor: {minor}개")