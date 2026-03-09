# generate.py

# pandas 라이브러리를 pd라는 별명으로 불러와요
# pandas = 파이썬에서 엑셀처럼 데이터 다루는 도구
import pandas as pd

# =============================================
# ① 데이터 준비
# =============================================

# 테스트 결과를 딕셔너리 리스트로 만들어요
# 딕셔너리 = {"키": 값} 형태
# 리스트 = [] 안에 여러 개 담기
test_results = [
    {"id": "TC-001", "name": "로그인 성공",      "result": "PASS", "time": 1.2},
    {"id": "TC-002", "name": "로그인 실패 처리",  "result": "PASS", "time": 0.8},
    {"id": "TC-003", "name": "비밀번호 변경",     "result": "FAIL", "time": 2.1},
    {"id": "TC-004", "name": "회원가입",          "result": "PASS", "time": 1.5},
    {"id": "TC-005", "name": "로그아웃",          "result": "PASS", "time": 0.5},
    {"id": "TC-006", "name": "상품 검색",         "result": "FAIL", "time": 3.2},
    {"id": "TC-007", "name": "장바구니 추가",     "result": "PASS", "time": 1.1},
    {"id": "TC-008", "name": "결제 처리",         "result": "FAIL", "time": 4.5},
    {"id": "TC-009", "name": "주문 내역 조회",    "result": "PASS", "time": 0.9},
    {"id": "TC-010", "name": "리뷰 작성",         "result": "PASS", "time": 1.3},
]

# =============================================
# ② Pandas로 분석
# =============================================

# 딕셔너리 리스트 → 엑셀 표(DataFrame) 형태로 변환
# df = DataFrame의 약자, 관례상 df로 많이 써요
df = pd.DataFrame(test_results)

# len() = 길이(개수)를 세는 함수
# df 전체 행 개수 = 테스트 총 개수
total = len(df)

# df['result'] == 'PASS' → result 열에서 PASS인 것만 True/False로 변환
# df[True/False] → True인 행만 남겨요
# len() → 그 개수를 세요
passed = len(df[df['result'] == 'PASS'])

# 실패 개수 = 전체 - 통과
failed = len(df[df['result'] == 'FAIL'])

# 통과율 계산
# passed / total = 0.7 → * 100 = 70.0
# round(숫자, 1) = 소수점 1자리까지만 보여줘요
pass_rate = round((passed / total) * 100, 1)

# df['time'] → time 열 전체
# .mean() → 평균 계산
# round(숫자, 2) = 소수점 2자리까지
avg_time = round(df['time'].mean(), 2)

# =============================================
# ③ 표 행(tr) 자동 생성
# =============================================

# rows = 나중에 HTML 표 안에 넣을 내용
# 지금은 빈 문자열로 시작해요
rows = ""

# df.iterrows() = 표의 각 행을 하나씩 꺼내줘요
# _ = 행 번호 (안 쓸 거라서 _로 무시)
# row = 그 행의 데이터 (id, name, result, time)
for _, row in df.iterrows():

    # result가 PASS면 초록 뱃지, FAIL이면 빨간 뱃지
    if row['result'] == 'PASS':
        badge = '<span class="pass">PASS</span>'
    else:
        badge = '<span class="fail">FAIL</span>'

    # f""" """ = 여러 줄 문자열 + 변수 끼워넣기
    # {row['id']} → 파이썬 변수를 HTML 안에 자동으로 넣어줘요
    # += 는 rows에 계속 이어붙이기
    rows += f"""
      <tr>
        <td>{row['id']}</td>
        <td>{row['name']}</td>
        <td>{badge}</td>
        <td>{row['time']}초</td>
      </tr>"""

# =============================================
# ④ HTML 파일 전체 생성
# =============================================

# f""" """ 로 HTML 코드를 통째로 문자열로 만들어요
# {변수} 부분에 파이썬이 계산한 값이 자동으로 들어가요
# CSS에서 { } 는 {{ }} 로 써야해요 (파이썬이랑 헷갈리지 않게)
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>QA 리포트</title>
  <style>
    body {{ font-family: sans-serif; padding: 40px; background: #f8f9fa; }}
    h1 {{ margin-bottom: 24px; }}
    .stats {{ display: flex; gap: 16px; margin-bottom: 32px; }}
    .stat {{ background: white; border: 1px solid #eee; border-radius: 10px; padding: 20px 28px; text-align: center; }}
    .stat-num {{ font-size: 36px; font-weight: 700; }}
    .stat-label {{ font-size: 12px; color: #999; margin-top: 4px; }}
    .green {{ color: #16a34a; }}
    .red   {{ color: #dc2626; }}
    .blue  {{ color: #2563eb; }}
    .gold  {{ color: #d97706; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; }}
    th {{ background: #1e1e1e; color: white; padding: 12px 16px; text-align: left; }}
    td {{ padding: 12px 16px; border-bottom: 1px solid #f0f0f0; }}
    .pass {{ background: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 4px; font-size: 12px; }}
    .fail {{ background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>

  <h1>🐛 QA 테스트 리포트</h1>

   <!-- 파이썬이 계산한 통계가 아래 자리에 자동으로 들어가요 -->
  <div class="stats">
    <div class="stat">
      <div class="stat-num">{total}</div>
      <div class="stat-label">전체 테스트</div>
    </div>
    <div class="stat">
      <div class="stat-num green">{passed}</div>
      <div class="stat-label">통과 ✅</div>
    </div>
    <div class="stat">
      <div class="stat-num red">{failed}</div>
      <div class="stat-label">실패 ❌</div>
    </div>
    <div class="stat">
      <div class="stat-num gold">{pass_rate}%</div>
      <div class="stat-label">통과율</div>
    </div>
    <div class="stat">
      <div class="stat-num blue">{avg_time}s</div>
      <div class="stat-label">평균 시간</div>
    </div>
  </div>

    <!-- 위에서 만든 표 행들이 여기 들어가요 -->
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>테스트명</th>
        <th>결과</th>
        <th>실행시간</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

</body>
</html>"""

# =============================================
# ⑤ 파일로 저장
# =============================================

# open("파일명", "w") = 파일을 쓰기 모드로 열어요
# "w" = write (쓰기)
# encoding="utf-8" = 한글 깨짐 방지
# as f = 이 파일을 f라는 이름으로 부를게요
with open("qa_report.html", "w", encoding="utf-8") as f:
    # f.write() = 파일에 내용을 써요
    f.write(html)

# 완료 메시지 출력
print("✅ qa_report.html 생성 완료!")
print(f"   전체: {total}개 | 통과: {passed}개 | 실패: {failed}개 | 통과율: {pass_rate}%")
