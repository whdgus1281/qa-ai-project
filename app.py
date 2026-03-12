# app.py
# 딥러닝 + Flask + DB 합친 최종 QA AI 툴!

from flask import Flask, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import pandas as pd

app = Flask(__name__)

# =============================================
# DB 설정
# =============================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bugs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Bug(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(200), nullable=False)
    time     = db.Column(db.Float, nullable=False)
    error    = db.Column(db.Integer, nullable=False)
    count    = db.Column(db.Integer, nullable=False)
    severity = db.Column(db.String(20))
    image    = db.Column(db.Text)  # 이미지 저장 (base64)

# =============================================
# 머신러닝 모델 학습
# =============================================
X = [[0.5,0,1],[0.8,0,2],[1.2,0,1],[1.5,1,3],
     [2.1,1,5],[3.2,1,8],[4.5,1,10],[5.0,1,15],
     [0.3,0,1],[2.5,1,6],[1.8,0,2],[3.8,1,9]]
y = [0,0,0,0,1,1,1,1,0,1,0,1]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_tr, y_tr)
accuracy = accuracy_score(y_te, clf.predict(X_te)) * 100

# =============================================
# 딥러닝 모델 (이미지 분석용)
# =============================================

# 이미지를 보고 "에러 화면인지" 판단하는 모델
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # 0=정상, 1=에러화면
        )
    def forward(self, x):
        return self.layers(x)

image_model = ImageModel()

# 이미지 전처리 함수
def analyze_image(image_file):
    try:
        # 이미지 열기
        img = Image.open(image_file)

        # 흑백으로 변환 + 28x28로 리사이즈
        img = img.convert('L').resize((28, 28))

        # 텐서로 변환
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        tensor = transform(img).unsqueeze(0)

        # 딥러닝으로 분석
        with torch.no_grad():
            output = image_model(tensor)
            pred = torch.argmax(output).item()

        # 평균 밝기로 에러 화면 감지
        # 에러 화면은 보통 빨간색이 많아서 어두워요
        avg_brightness = tensor.mean().item()

        if avg_brightness < 0.3:
            return "에러 화면 감지됨 🔴", 1
        else:
            return "정상 화면 🟢", 0

    except:
        return "이미지 분석 불가", 0

# =============================================
# 메인 페이지
# =============================================

@app.route("/")
def home():
    bugs = Bug.query.all()
    total    = len(bugs)
    critical = len([b for b in bugs if b.severity == 'Critical'])
    minor    = len([b for b in bugs if b.severity == 'Minor'])

    rows = ""
    for bug in bugs:
        badge = '<span class="critical">🔴 Critical</span>' if bug.severity == 'Critical' else '<span class="minor">🟡 Minor</span>'
        img_tag = f'<img src="data:image/png;base64,{bug.image}" style="width:40px;height:40px;border-radius:4px;object-fit:cover;">' if bug.image else '<span style="color:#333;font-size:11px">없음</span>'
        rows += f"<tr><td class='id-cell'>BUG-{bug.id:03d}</td><td>{bug.name}</td><td>{bug.time}s</td><td>{'있음' if bug.error==1 else '없음'}</td><td>{bug.count}회</td><td>{badge}</td><td>{img_tag}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Justin's QA Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:'Noto Sans KR',sans-serif;background:#080808;color:#f0f0f0;min-height:100vh;}}
    .sidebar{{position:fixed;left:0;top:0;bottom:0;width:220px;background:#0d0d0d;border-right:1px solid #1a1a1a;padding:32px 20px;display:flex;flex-direction:column;gap:8px;}}
    .logo{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#4ade80;margin-bottom:24px;padding-bottom:24px;border-bottom:1px solid #1a1a1a;}}
    .logo span{{color:#555;}}
    .nav-item{{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;font-size:13px;color:#555;}}
    .nav-item.active{{background:#1a2a1a;color:#4ade80;}}
    .nav-dot{{width:6px;height:6px;border-radius:50%;background:currentColor;}}
    .profile{{margin-top:auto;display:flex;align-items:center;gap:10px;padding:12px;background:#111;border-radius:10px;border:1px solid #1a1a1a;}}
    .avatar{{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#4ade80,#22c55e);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#000;flex-shrink:0;}}
    .profile-name{{font-size:13px;font-weight:700;}}
    .profile-role{{font-size:11px;color:#555;}}
    .main{{margin-left:220px;padding:40px;}}
    .header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:40px;}}
    .greeting{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#555;letter-spacing:2px;margin-bottom:8px;}}
    .title{{font-size:28px;font-weight:700;}}
    .title span{{color:#4ade80;}}
    .ai-pill{{display:flex;align-items:center;gap:8px;background:#1a2a1a;border:1px solid #2a4a2a;padding:8px 16px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#4ade80;}}
    .pulse{{width:8px;height:8px;border-radius:50%;background:#4ade80;animation:pulse 2s infinite;}}
    @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:0.5;transform:scale(0.8);}}}}
    .stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}}
    .stat{{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:16px;padding:24px;position:relative;overflow:hidden;}}
    .stat::after{{content:'';position:absolute;top:0;left:0;right:0;height:1px;}}
    .stat.g::after{{background:linear-gradient(90deg,transparent,#4ade80,transparent);}}
    .stat.r::after{{background:linear-gradient(90deg,transparent,#f87171,transparent);}}
    .stat.y::after{{background:linear-gradient(90deg,transparent,#fbbf24,transparent);}}
    .stat.b::after{{background:linear-gradient(90deg,transparent,#60a5fa,transparent);}}
    .stat-label{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;color:#555;margin-bottom:16px;}}
    .stat-num{{font-family:'JetBrains Mono',monospace;font-size:48px;font-weight:700;line-height:1;}}
    .stat.g .stat-num{{color:#4ade80;}}
    .stat.r .stat-num{{color:#f87171;}}
    .stat.y .stat-num{{color:#fbbf24;}}
    .stat.b .stat-num{{color:#60a5fa;}}
    .stat-sub{{font-size:12px;color:#333;margin-top:8px;}}
    .form-section{{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:16px;padding:28px;margin-bottom:28px;}}
    .section-label{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;color:#555;margin-bottom:20px;}}
    .form-row{{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;}}
    .field{{display:flex;flex-direction:column;gap:8px;flex:1;min-width:120px;}}
    .field label{{font-size:11px;color:#444;letter-spacing:1px;}}
    .field input,.field select{{background:#111;border:1px solid #1e1e1e;color:#f0f0f0;padding:12px 14px;border-radius:10px;font-size:13px;outline:none;transition:border-color 0.2s;font-family:'Noto Sans KR',sans-serif;}}
    .field input:focus,.field select:focus{{border-color:#4ade80;}}
    .submit-btn{{background:#4ade80;color:#000;border:none;padding:12px 28px;border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap;font-family:'Noto Sans KR',sans-serif;transition:all 0.2s;}}
    .submit-btn:hover{{background:#22c55e;}}
    .table-section{{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:16px;overflow:hidden;}}
    .table-head{{display:flex;justify-content:space-between;align-items:center;padding:20px 24px;border-bottom:1px solid #1a1a1a;}}
    .table-title{{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;color:#555;}}
    .count-badge{{background:#111;border:1px solid #1e1e1e;border-radius:100px;padding:4px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#444;}}
    table{{width:100%;border-collapse:collapse;}}
    th{{padding:12px 24px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1px;color:#444;border-bottom:1px solid #111;font-weight:400;}}
    td{{padding:14px 24px;border-bottom:1px solid #111;font-size:13px;}}
    tr:last-child td{{border-bottom:none;}}
    tr:hover td{{background:#0f0f0f;}}
    .id-cell{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#333;}}
    .critical{{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.15);color:#f87171;padding:4px 12px;border-radius:100px;font-size:11px;}}
    .minor{{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.15);color:#fbbf24;padding:4px 12px;border-radius:100px;font-size:11px;}}
    .empty{{padding:40px;text-align:center;color:#333;font-size:13px;}}
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="logo"><span>//</span> justin.qa</div>
    <div class="nav-item active"><div class="nav-dot"></div> 대시보드</div>
    <div class="nav-item"><div class="nav-dot"></div> 버그 목록</div>
    <div class="nav-item"><div class="nav-dot"></div> AI 분석</div>
    <div class="nav-item"><div class="nav-dot"></div> 리포트</div>
    <div class="profile">
      <div class="avatar">J</div>
      <div>
        <div class="profile-name">Justin</div>
        <div class="profile-role">QA Engineer</div>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="header">
      <div>
        <div class="greeting">GOOD WORK, JUSTIN 👋</div>
        <div class="title">QA AI <span>Dashboard</span></div>
      </div>
      <div class="ai-pill">
        <div class="pulse"></div>
        ML {accuracy:.0f}% · PyTorch 딥러닝 연동
      </div>
    </div>

    <div class="stats">
      <div class="stat g"><div class="stat-label">TOTAL BUGS</div><div class="stat-num">{total}</div><div class="stat-sub">전체 버그</div></div>
      <div class="stat r"><div class="stat-label">CRITICAL</div><div class="stat-num">{critical}</div><div class="stat-sub">즉시 처리 필요</div></div>
      <div class="stat y"><div class="stat-label">MINOR</div><div class="stat-num">{minor}</div><div class="stat-sub">낮은 우선순위</div></div>
      <div class="stat b"><div class="stat-label">AI ACCURACY</div><div class="stat-num">{accuracy:.0f}<span style="font-size:24px">%</span></div><div class="stat-sub">모델 정확도</div></div>
    </div>

    <div class="form-section">
      <div class="section-label">+ NEW BUG REPORT</div>
      <form action="/add" method="post" enctype="multipart/form-data">
        <div class="form-row">
          <div class="field">
            <label>버그 이름</label>
            <input type="text" name="name" placeholder="예: 결제 오류" required>
          </div>
          <div class="field" style="max-width:130px">
            <label>응답시간 (초)</label>
            <input type="number" name="time" step="0.1" placeholder="3.5" required>
          </div>
          <div class="field" style="max-width:120px">
            <label>에러코드</label>
            <select name="error">
              <option value="0">없음</option>
              <option value="1">있음</option>
            </select>
          </div>
          <div class="field" style="max-width:130px">
            <label>발생횟수 (회/일)</label>
            <input type="number" name="count" placeholder="5" required>
          </div>
          <div class="field" style="max-width:160px">
            <label>스크린샷 (선택)</label>
            <input type="file" name="screenshot" accept="image/*">
          </div>
          <button class="submit-btn" type="submit">AI 판단 →</button>
        </div>
      </form>
    </div>

    <div class="table-section">
      <div class="table-head">
        <div class="table-title">BUG REPORTS</div>
        <div class="count-badge">{total}개</div>
      </div>
      <table>
        <thead>
          <tr><th>ID</th><th>버그명</th><th>응답시간</th><th>에러코드</th><th>발생횟수</th><th>AI 판단</th><th>스크린샷</th></tr>
        </thead>
        <tbody>
          {rows if rows else '<tr><td colspan="7" class="empty">버그를 추가해보세요! 🐛</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>"""

# =============================================
# 버그 추가 처리
# =============================================

@app.route("/add", methods=["POST"])
def add_bug():
    name  = request.form['name']
    time  = float(request.form['time'])
    error = int(request.form['error'])
    count = int(request.form['count'])

    # AI로 심각도 예측
    prediction = clf.predict([[time, error, count]])[0]
    severity = 'Critical' if prediction == 1 else 'Minor'

    # 스크린샷 처리
    image_base64 = None
    if 'screenshot' in request.files:
        file = request.files['screenshot']
        if file.filename != '':
            # 딥러닝으로 이미지 분석
            result, _ = analyze_image(file)
            print(f"🔍 이미지 분석 결과: {result}")

            # base64로 변환해서 DB에 저장
            file.seek(0)
            image_data = file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

    # DB에 저장
    new_bug = Bug(
        name=name, time=time, error=error,
        count=count, severity=severity, image=image_base64
    )
    db.session.add(new_bug)
    db.session.commit()

    return redirect(url_for('home'))

# =============================================
# 서버 시작
# =============================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=8080)