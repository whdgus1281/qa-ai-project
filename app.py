# app.py
# QA 자동화 전체 시스템 최종 완성!

from flask import Flask, request, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import base64, io, json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bugs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 한글 폰트
pdfmetrics.registerFont(TTFont('Korean', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
pdfmetrics.registerFont(TTFont('Korean-Bold', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))

# =============================================
# DB 모델
# =============================================
class Bug(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(200), nullable=False)
    time     = db.Column(db.Float, nullable=False)
    error    = db.Column(db.Integer, nullable=False)
    count    = db.Column(db.Integer, nullable=False)
    severity = db.Column(db.String(20))
    image    = db.Column(db.Text)

# =============================================
# 머신러닝 모델
# =============================================
X_ml = [[0.5,0,1],[0.8,0,2],[1.2,0,1],[1.5,1,3],
         [2.1,1,5],[3.2,1,8],[4.5,1,10],[5.0,1,15],
         [0.3,0,1],[2.5,1,6],[1.8,0,2],[3.8,1,9]]
y_ml = [0,0,0,0,1,1,1,1,0,1,0,1]
X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_tr, y_tr)
ml_accuracy = accuracy_score(y_te, clf.predict(X_te)) * 100

# =============================================
# NLP 모델
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
nlp_labels = [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]
vectorizer = TfidfVectorizer()
X_nlp = vectorizer.fit_transform(bug_texts)
X_ntr, X_nte, y_ntr, y_nte = train_test_split(X_nlp, nlp_labels, test_size=0.2, random_state=42)
nlp_clf = DecisionTreeClassifier()
nlp_clf.fit(X_ntr, y_ntr)
nlp_accuracy = accuracy_score(y_nte, nlp_clf.predict(X_nte)) * 100

# =============================================
# 딥러닝 이미지 모델
# =============================================
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.layers(x)

image_model = ImageModel()

def analyze_image(image_file):
    try:
        img = Image.open(image_file).convert('L').resize((28, 28))
        tensor = transforms.ToTensor()(img).unsqueeze(0)
        avg_brightness = tensor.mean().item()
        if avg_brightness < 0.3:
            return "에러 화면 감지됨 🔴"
        else:
            return "정상 화면 🟢"
    except:
        return "분석 불가"

print(f"✅ ML 정확도: {ml_accuracy:.0f}% | NLP 정확도: {nlp_accuracy:.0f}%")

# =============================================
# 메인 대시보드
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
        img_tag = f'<img src="data:image/png;base64,{bug.image}" style="width:36px;height:36px;border-radius:4px;object-fit:cover;">' if bug.image else '<span style="color:#333;font-size:11px">없음</span>'
        rows += f"<tr><td class='id-cell'>BUG-{bug.id:03d}</td><td>{bug.name}</td><td>{bug.time}s</td><td>{'있음' if bug.error==1 else '없음'}</td><td>{bug.count}회</td><td>{badge}</td><td>{img_tag}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Justin's QA System</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:'Noto Sans KR',sans-serif;background:#080808;color:#f0f0f0;min-height:100vh;}}
    .sidebar{{position:fixed;left:0;top:0;bottom:0;width:220px;background:#0d0d0d;border-right:1px solid #1a1a1a;padding:32px 20px;display:flex;flex-direction:column;gap:8px;}}
    .logo{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#4ade80;margin-bottom:24px;padding-bottom:24px;border-bottom:1px solid #1a1a1a;}}
    .logo span{{color:#555;}}
    .nav-item{{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;font-size:13px;color:#555;cursor:pointer;text-decoration:none;}}
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
    .header-btns{{display:flex;gap:10px;}}
    .ai-pill{{display:flex;align-items:center;gap:8px;background:#1a2a1a;border:1px solid #2a4a2a;padding:8px 16px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#4ade80;text-decoration:none;}}
    .report-btn{{display:flex;align-items:center;gap:8px;background:#4ade80;border:none;padding:8px 16px;border-radius:100px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#000;font-weight:700;cursor:pointer;text-decoration:none;}}
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
    .nlp-section{{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:16px;padding:28px;margin-bottom:28px;}}
    .section-label{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;color:#555;margin-bottom:16px;}}
    .nlp-row{{display:flex;gap:12px;}}
    .nlp-input{{flex:1;background:#111;border:1px solid #1e1e1e;color:#f0f0f0;padding:14px 16px;border-radius:10px;font-size:13px;outline:none;font-family:'Noto Sans KR',sans-serif;transition:border-color 0.2s;}}
    .nlp-input:focus{{border-color:#4ade80;}}
    .nlp-btn{{background:#4ade80;color:#000;border:none;padding:14px 24px;border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;font-family:'Noto Sans KR',sans-serif;white-space:nowrap;}}
    .nlp-result{{margin-top:12px;padding:14px 16px;border-radius:10px;font-size:13px;display:none;}}
    .nlp-critical{{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);color:#f87171;}}
    .nlp-minor{{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);color:#fbbf24;}}
    .form-section{{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:16px;padding:28px;margin-bottom:28px;}}
    .form-row{{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;}}
    .field{{display:flex;flex-direction:column;gap:8px;flex:1;min-width:120px;}}
    .field label{{font-size:11px;color:#444;letter-spacing:1px;}}
    .field input,.field select{{background:#111;border:1px solid #1e1e1e;color:#f0f0f0;padding:12px 14px;border-radius:10px;font-size:13px;outline:none;font-family:'Noto Sans KR',sans-serif;}}
    .field input:focus,.field select:focus{{border-color:#4ade80;}}
    .submit-btn{{background:#4ade80;color:#000;border:none;padding:12px 28px;border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;font-family:'Noto Sans KR',sans-serif;}}
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
    <a class="nav-item active" href="/"><div class="nav-dot"></div> 대시보드</a>
    <a class="nav-item" href="/report"><div class="nav-dot"></div> PDF 리포트</a>
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
        <div class="title">QA AI <span>System</span></div>
      </div>
      <div class="header-btns">
        <div class="ai-pill"><div class="pulse"></div> ML {ml_accuracy:.0f}% · NLP {nlp_accuracy:.0f}%</div>
        <a class="report-btn" href="/report">📄 PDF 리포트</a>
      </div>
    </div>

    <div class="stats">
      <div class="stat g"><div class="stat-label">TOTAL BUGS</div><div class="stat-num">{total}</div><div class="stat-sub">전체 버그</div></div>
      <div class="stat r"><div class="stat-label">CRITICAL</div><div class="stat-num">{critical}</div><div class="stat-sub">즉시 처리 필요</div></div>
      <div class="stat y"><div class="stat-label">MINOR</div><div class="stat-num">{minor}</div><div class="stat-sub">낮은 우선순위</div></div>
      <div class="stat b"><div class="stat-label">AI ACCURACY</div><div class="stat-num">{ml_accuracy:.0f}<span style="font-size:24px">%</span></div><div class="stat-sub">모델 정확도</div></div>
    </div>

    <div class="nlp-section">
      <div class="section-label">🧠 NLP 실시간 분석</div>
      <div class="nlp-row">
        <input class="nlp-input" id="nlpText" placeholder="버그 내용을 입력하면 AI가 즉시 판단해요...">
        <button class="nlp-btn" onclick="nlpAnalyze()">AI 판단 →</button>
      </div>
      <div class="nlp-result" id="nlpResult"></div>
    </div>

    <div class="form-section">
      <div class="section-label">+ NEW BUG REPORT</div>
      <form action="/add" method="post" enctype="multipart/form-data">
        <div class="form-row">
          <div class="field"><label>버그 이름</label><input type="text" name="name" placeholder="예: 결제 오류" required></div>
          <div class="field" style="max-width:130px"><label>응답시간 (초)</label><input type="number" name="time" step="0.1" placeholder="3.5" required></div>
          <div class="field" style="max-width:120px"><label>에러코드</label><select name="error"><option value="0">없음</option><option value="1">있음</option></select></div>
          <div class="field" style="max-width:130px"><label>발생횟수</label><input type="number" name="count" placeholder="5" required></div>
          <div class="field" style="max-width:160px"><label>스크린샷 (선택)</label><input type="file" name="screenshot" accept="image/*"></div>
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
        <thead><tr><th>ID</th><th>버그명</th><th>응답시간</th><th>에러</th><th>발생횟수</th><th>AI 판단</th><th>스크린샷</th></tr></thead>
        <tbody>{rows if rows else '<tr><td colspan="7" class="empty">버그를 추가해보세요! 🐛</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <script>
    async function nlpAnalyze() {{
      const text = document.getElementById('nlpText').value;
      if (!text.trim()) return;
      const res = await fetch('/nlp', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{text: text}})
      }});
      const data = await res.json();
      const el = document.getElementById('nlpResult');
      el.className = 'nlp-result ' + (data.severity === 'Critical' ? 'nlp-critical' : 'nlp-minor');
      el.style.display = 'block';
      el.textContent = data.severity === 'Critical'
        ? `🔴 Critical — 즉시 처리 필요! (확신도 ${{data.confidence}}%)`
        : `🟡 Minor — 낮은 우선순위 (확신도 ${{data.confidence}}%)`;
    }}
  </script>
</body>
</html>"""

# =============================================
# 버그 추가
# =============================================
@app.route("/add", methods=["POST"])
def add_bug():
    name  = request.form['name']
    time  = float(request.form['time'])
    error = int(request.form['error'])
    count = int(request.form['count'])

    prediction = clf.predict([[time, error, count]])[0]
    severity = 'Critical' if prediction == 1 else 'Minor'

    image_base64 = None
    if 'screenshot' in request.files:
        file = request.files['screenshot']
        if file.filename != '':
            result = analyze_image(file)
            print(f"🔍 이미지 분석: {result}")
            file.seek(0)
            image_base64 = base64.b64encode(file.read()).decode('utf-8')

    db.session.add(Bug(name=name, time=time, error=error,
                       count=count, severity=severity, image=image_base64))
    db.session.commit()
    return redirect(url_for('home'))

# =============================================
# NLP API
# =============================================
@app.route("/nlp", methods=["POST"])
def nlp_analyze():
    text = request.get_json().get("text", "")
    X_new = vectorizer.transform([text])
    pred = nlp_clf.predict(X_new)[0]
    conf = round(max(nlp_clf.predict_proba(X_new)[0]) * 100, 1)
    return {"severity": "Critical" if pred == 1 else "Minor", "confidence": conf}

# =============================================
# PDF 리포트 자동 생성 + 다운로드
# =============================================
@app.route("/report")
def generate_report():
    bugs = Bug.query.all()
    if not bugs:
        return "<h2 style='font-family:sans-serif;padding:40px;color:#555'>버그 데이터가 없어요! 먼저 버그를 추가해주세요 🐛</h2>"

    total    = len(bugs)
    critical = len([b for b in bugs if b.severity == 'Critical'])
    minor    = len([b for b in bugs if b.severity == 'Minor'])

    path = "/tmp/qa_report.pdf"
    doc = SimpleDocTemplate(path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    title_style   = ParagraphStyle('T', fontName='Korean-Bold', fontSize=22,
                                   textColor=colors.HexColor('#111111'), spaceAfter=6)
    sub_style     = ParagraphStyle('S', fontName='Korean', fontSize=10,
                                   textColor=colors.HexColor('#888888'), spaceAfter=4)
    section_style = ParagraphStyle('SEC', fontName='Korean-Bold', fontSize=13,
                                   textColor=colors.HexColor('#111111'), spaceBefore=16, spaceAfter=8)
    footer_style  = ParagraphStyle('F', fontName='Korean', fontSize=8,
                                   textColor=colors.HexColor('#aaaaaa'), alignment=1)

    content = []
    content.append(Paragraph("QA AI 자동화 리포트", title_style))
    content.append(Paragraph(f"Justin's QA AI 시스템  |  {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}", sub_style))
    content.append(Spacer(1, 0.4*cm))
    content.append(Table([['']], colWidths=[17*cm],
        style=TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.HexColor('#eeeeee'))])))
    content.append(Spacer(1, 0.4*cm))

    content.append(Paragraph("📊 요약", section_style))
    summary_data = [
        ['전체 버그', 'Critical', 'Minor', 'AI 정확도'],
        [str(total), str(critical), str(minor), f"{ml_accuracy:.0f}%"]
    ]
    summary_table = Table(summary_data, colWidths=[4.25*cm]*4)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#111111')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Korean-Bold'),
        ('FONTNAME',(0,1),(-1,1),'Korean-Bold'),
        ('FONTSIZE',(0,0),(-1,0),9),
        ('FONTSIZE',(0,1),(-1,1),16),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWHEIGHT',(0,0),(-1,0),36),
        ('ROWHEIGHT',(0,1),(-1,1),55),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#f9f9f9')),
        ('TEXTCOLOR',(1,1),(1,1),colors.HexColor('#ef4444')),
        ('TEXTCOLOR',(2,1),(2,1),colors.HexColor('#f59e0b')),
        ('TEXTCOLOR',(3,1),(3,1),colors.HexColor('#22c55e')),
        ('BOX',(0,0),(-1,-1),1,colors.HexColor('#eeeeee')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#eeeeee')),
    ]))
    content.append(summary_table)
    content.append(Spacer(1, 0.5*cm))

    content.append(Paragraph("🐛 버그 목록", section_style))
    table_data = [['ID', '버그명', '응답시간', '에러', '발생횟수', '심각도']]
    for bug in bugs:
        table_data.append([
            f"BUG-{bug.id:03d}", bug.name, f"{bug.time}s",
            '있음' if bug.error==1 else '없음',
            f"{bug.count}회", bug.severity
        ])
    bug_table = Table(table_data, colWidths=[2.2*cm, 5*cm, 2*cm, 1.8*cm, 1.8*cm, 4.2*cm])
    bug_table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#111111')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Korean-Bold'),
        ('FONTNAME',(0,1),(-1,-1),'Korean'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWHEIGHT',(0,0),(-1,-1),24),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#fafafa')]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#eeeeee')),
        ('BOX',(0,0),(-1,-1),1,colors.HexColor('#eeeeee')),
    ]))
    for i, bug in enumerate(bugs, 1):
        if bug.severity == 'Critical':
            bug_table.setStyle(TableStyle([
                ('TEXTCOLOR',(5,i),(5,i),colors.HexColor('#ef4444')),
                ('FONTNAME',(5,i),(5,i),'Korean-Bold'),
            ]))
        else:
            bug_table.setStyle(TableStyle([
                ('TEXTCOLOR',(5,i),(5,i),colors.HexColor('#f59e0b')),
            ]))
    content.append(bug_table)
    content.append(Spacer(1, 0.5*cm))

    content.append(Table([['']], colWidths=[17*cm],
        style=TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.HexColor('#eeeeee'))])))
    content.append(Spacer(1, 0.3*cm))
    content.append(Paragraph(
        f"본 리포트는 Justin's QA AI 시스템이 자동 생성했습니다  |  AI 정확도: {ml_accuracy:.0f}%",
        footer_style
    ))

    doc.build(content)
    return send_file(path, as_attachment=True, download_name="qa_report.pdf")

# =============================================
# 서버 시작
# =============================================
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=8080)
