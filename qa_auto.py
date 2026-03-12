# qa_auto.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 한글 폰트 등록
pdfmetrics.registerFont(TTFont('Korean', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
pdfmetrics.registerFont(TTFont('Korean-Bold', '/System/Library/Fonts/Supplemental/AppleGothic.ttf'))

# =============================================
# ① AI 모델 학습
# =============================================
X = [[0.5,0,1],[0.8,0,2],[1.2,0,1],[1.5,1,3],
     [2.1,1,5],[3.2,1,8],[4.5,1,10],[5.0,1,15],
     [0.3,0,1],[2.5,1,6],[1.8,0,2],[3.8,1,9]]
y = [0,0,0,0,1,1,1,1,0,1,0,1]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_tr, y_tr)
accuracy = accuracy_score(y_te, model.predict(X_te)) * 100
print(f"✅ AI 모델 학습 완료! 정확도: {accuracy:.1f}%")

# =============================================
# ② 버그 데이터
# =============================================
bugs = [
    {"id":"BUG-001","name":"로그인 500 에러",    "time":4.2,"error":1,"count":12},
    {"id":"BUG-002","name":"메뉴 아이콘 깨짐",   "time":0.5,"error":0,"count":2},
    {"id":"BUG-003","name":"결제 타임아웃",      "time":5.1,"error":1,"count":20},
    {"id":"BUG-004","name":"오타 수정",           "time":0.3,"error":0,"count":1},
    {"id":"BUG-005","name":"검색 느림",           "time":3.1,"error":1,"count":7},
    {"id":"BUG-006","name":"버튼 색상 오류",      "time":0.4,"error":0,"count":3},
    {"id":"BUG-007","name":"회원가입 오류",       "time":3.8,"error":1,"count":9},
    {"id":"BUG-008","name":"이미지 로딩 실패",    "time":2.8,"error":1,"count":6},
]

# =============================================
# ③ AI로 자동 분류
# =============================================
df = pd.DataFrame(bugs)
X_new = df[['time','error','count']].values.tolist()
df['severity'] = ['Critical' if p==1 else 'Minor' for p in model.predict(X_new)]
df['priority_score'] = df['time'] * df['count']
df = df.sort_values('priority_score', ascending=False)

total    = len(df)
critical = len(df[df['severity'] == 'Critical'])
minor    = len(df[df['severity'] == 'Minor'])
avg_time = df['time'].mean()
top_bug  = df.iloc[0]

print(f"\n📊 통계: 전체 {total} | Critical {critical} | Minor {minor}")
print(f"🚨 최우선 버그: {top_bug['name']}")

# =============================================
# ④ PDF 리포트 자동 생성
# =============================================
print("\n📄 PDF 리포트 생성 중...")

doc = SimpleDocTemplate(
    "qa_report.pdf",
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

# 스타일
title_style = ParagraphStyle('T', fontName='Korean-Bold', fontSize=22,
    textColor=colors.HexColor('#111111'), spaceAfter=6)
sub_style = ParagraphStyle('S', fontName='Korean', fontSize=10,
    textColor=colors.HexColor('#888888'), spaceAfter=4)
section_style = ParagraphStyle('SEC', fontName='Korean-Bold', fontSize=13,
    textColor=colors.HexColor('#111111'), spaceBefore=16, spaceAfter=8)
footer_style = ParagraphStyle('F', fontName='Korean', fontSize=8,
    textColor=colors.HexColor('#aaaaaa'), alignment=1)

content = []

# 제목
content.append(Paragraph("QA AI 자동화 리포트", title_style))
content.append(Paragraph(f"Justin's QA AI 시스템  |  {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}", sub_style))
content.append(Spacer(1, 0.4*cm))

# 구분선
content.append(Table([['']], colWidths=[17*cm],
    style=TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.HexColor('#eeeeee'))])))
content.append(Spacer(1, 0.4*cm))

# 요약
content.append(Paragraph("📊 요약", section_style))
summary_data = [
    ['전체 버그', 'Critical', 'Minor', 'AI 정확도', '평균 응답시간'],
    [str(total), str(critical), str(minor), f"{accuracy:.0f}%", f"{avg_time:.2f}s"]
]
summary_table = Table(summary_data, colWidths=[3.2*cm, 3.2*cm, 3.2*cm, 3.7*cm, 3.7*cm])
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
    ('FONTSIZE',(0,1),(-1,1),14),
    ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#f9f9f9')),
    ('TEXTCOLOR',(1,1),(1,1),colors.HexColor('#ef4444')),
    ('TEXTCOLOR',(2,1),(2,1),colors.HexColor('#f59e0b')),
    ('TEXTCOLOR',(3,1),(3,1),colors.HexColor('#22c55e')),
    ('BOX',(0,0),(-1,-1),1,colors.HexColor('#eeeeee')),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#eeeeee')),
]))
content.append(summary_table)
content.append(Spacer(1, 0.5*cm))

# 버그 목록
content.append(Paragraph("🐛 버그 목록 (AI 우선순위 정렬)", section_style))
table_data = [['ID', '버그명', '응답시간', '에러', '발생횟수', '심각도', '점수']]
for _, row in df.iterrows():
    table_data.append([
        row['id'], row['name'], f"{row['time']}s",
        '있음' if row['error']==1 else '없음',
        f"{row['count']}회", row['severity'], f"{row['priority_score']:.1f}"
    ])

bug_table = Table(table_data, colWidths=[2.1*cm, 4.3*cm, 2*cm, 1.6*cm, 1.8*cm, 2.4*cm, 1.8*cm])
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
for i, (_, row) in enumerate(df.iterrows(), 1):
    if row['severity'] == 'Critical':
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

# 최우선 버그
content.append(Paragraph("🚨 최우선 처리 버그", section_style))
top_data = [
    ['버그 ID', '버그명', '심각도', '우선순위 점수'],
    [top_bug['id'], top_bug['name'], top_bug['severity'], f"{top_bug['priority_score']:.1f}"]
]
top_table = Table(top_data, colWidths=[3*cm, 6*cm, 3*cm, 5*cm])
top_table.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ef4444')),
    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Korean-Bold'),
    ('FONTNAME',(0,1),(-1,1),'Korean-Bold'),
    ('FONTSIZE',(0,0),(-1,-1),10),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('ROWHEIGHT',(0,0),(-1,-1),30),
    ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#fff5f5')),
    ('TEXTCOLOR',(0,1),(-1,1),colors.HexColor('#ef4444')),
    ('BOX',(0,0),(-1,-1),1,colors.HexColor('#ef4444')),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#ffcccc')),
]))
content.append(top_table)
content.append(Spacer(1, 0.5*cm))

# 푸터
content.append(Table([['']], colWidths=[17*cm],
    style=TableStyle([('LINEABOVE',(0,0),(-1,0),1,colors.HexColor('#eeeeee'))])))
content.append(Spacer(1, 0.3*cm))
content.append(Paragraph(
    f"본 리포트는 Justin's QA AI 시스템이 자동 생성했습니다  |  AI 정확도: {accuracy:.0f}%",
    footer_style
))

doc.build(content)
print("✅ qa_report.pdf 생성 완료!")
