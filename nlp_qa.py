# nlp_qa.py
# NLP + Flask 연결!
# 버그 텍스트 입력 → AI가 자동 판단

from flask import Flask, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =============================================
# ① 학습 데이터
# =============================================
bug_texts = [
    "로그인 페이지 500 서버 에러 발생",
    "결제 진행중 시스템 다운",
    "DB 연결 실패로 전체 서비스 중단",
    "회원가입 오류 데이터 저장 안됨",
    "API 응답 없음 타임아웃 에러",
    "서버 크래시 모든 사용자 접속 불가",
    "결제 오류 500에러 반복 발생",
    "로그아웃 안됨 세션 에러",
    "서버 다운 전체 접속 불가",
    "DB 에러 데이터 손실 발생",
    "버튼 색상이 디자인과 다름",
    "오타 수정 필요 메인 페이지",
    "아이콘 깨져보임 모바일에서",
    "폰트 크기 통일 필요",
    "여백 조정 필요 footer 부분",
    "이미지 살짝 흐릿하게 보임",
    "툴팁 문구 수정 필요",
    "메뉴 정렬 약간 틀어짐",
    "색상 변경 요청",
    "오타 하나 발견",
]
labels = [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]

# =============================================
# ② AI 학습
# =============================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(bug_texts)
X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_tr, y_tr)
accuracy = accuracy_score(y_te, model.predict(X_te)) * 100
print(f"✅ NLP AI 학습 완료! 정확도: {accuracy:.1f}%")

# =============================================
# ③ Flask 페이지
# =============================================
@app.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>NLP QA AI</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'Noto Sans KR',sans-serif; background:#080808; color:#f0f0f0; min-height:100vh; display:flex; align-items:center; justify-content:center; }
    .card { background:#0d0d0d; border:1px solid #1a1a1a; border-radius:20px; padding:48px; width:100%; max-width:600px; }
    .title { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:3px; color:#555; margin-bottom:12px; }
    h1 { font-size:28px; font-weight:700; margin-bottom:8px; }
    h1 span { color:#4ade80; }
    .sub { font-size:13px; color:#555; margin-bottom:40px; }
    textarea {
      width:100%; background:#111; border:1px solid #1e1e1e; color:#f0f0f0;
      padding:16px; border-radius:12px; font-size:14px; font-family:'Noto Sans KR',sans-serif;
      resize:none; outline:none; height:120px; transition:border-color 0.2s;
    }
    textarea:focus { border-color:#4ade80; }
    textarea::placeholder { color:#333; }
    .btn {
      width:100%; background:#4ade80; color:#000; border:none;
      padding:16px; border-radius:12px; font-size:14px; font-weight:700;
      cursor:pointer; margin-top:12px; font-family:'Noto Sans KR',sans-serif;
      transition:background 0.2s;
    }
    .btn:hover { background:#22c55e; }
    .result { margin-top:24px; padding:24px; border-radius:12px; display:none; }
    .critical { background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.2); }
    .minor { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2); }
    .result-label { font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:2px; margin-bottom:8px; }
    .result-text { font-size:24px; font-weight:700; }
    .critical .result-label { color:#f87171; }
    .critical .result-text { color:#f87171; }
    .minor .result-label { color:#fbbf24; }
    .minor .result-text { color:#fbbf24; }
    .confidence { font-size:12px; color:#555; margin-top:8px; }
    .accuracy { text-align:center; margin-top:32px; font-family:'JetBrains Mono',monospace; font-size:11px; color:#333; }
    .accuracy span { color:#4ade80; }
  </style>
</head>
<body>
  <div class="card">
    <div class="title">JUSTIN'S NLP QA AI</div>
    <h1>버그 <span>자동 분류</span></h1>
    <p class="sub">버그 내용을 입력하면 AI가 자동으로 심각도를 판단해요</p>

    <textarea id="bugText" placeholder="예: 로그인 페이지에서 500 에러가 계속 발생합니다"></textarea>
    <button class="btn" onclick="analyze()">AI 판단하기 →</button>

    <div class="result" id="result">
      <div class="result-label" id="resultLabel">AI 판단</div>
      <div class="result-text" id="resultText"></div>
      <div class="confidence" id="confidence"></div>
    </div>

    <div class="accuracy">AI 정확도: <span>""" + f"{accuracy:.0f}%" + """</span></div>
  </div>

  <script>
    async function analyze() {
      const text = document.getElementById('bugText').value;
      if (!text.trim()) return;

      const res = await fetch('/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
      });
      const data = await res.json();

      const result = document.getElementById('result');
      result.className = 'result ' + (data.severity === 'Critical' ? 'critical' : 'minor');
      result.style.display = 'block';

      document.getElementById('resultLabel').textContent = 'AI 판단 결과';
      document.getElementById('resultText').textContent =
        data.severity === 'Critical' ? '🔴 Critical — 즉시 처리 필요!' : '🟡 Minor — 낮은 우선순위';
      document.getElementById('confidence').textContent = `확신도: ${data.confidence}%`;
    }
  </script>
</body>
</html>"""

# =============================================
# ④ AI 분석 API
# =============================================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    X_new = vectorizer.transform([text])
    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    confidence = round(max(proba) * 100, 1)
    severity = "Critical" if prediction == 1 else "Minor"

    print(f"📝 입력: {text}")
    print(f"🤖 판단: {severity} ({confidence}%)")

    return {"severity": severity, "confidence": confidence}

# =============================================
# ⑤ 서버 실행
# =============================================
if __name__ == "__main__":
    app.run(debug=True, port=8081)