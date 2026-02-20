from __future__ import annotations

from datetime import date
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

PROJECT_DIR = Path('/Users/mokshagnaram/Desktop/Dog_Breed/DogBreedProject')
MD_PATH = PROJECT_DIR / 'Dog_Breed_Identification_Report.md'
PDF_PATH = PROJECT_DIR / 'Dog_Breed_Identification_Report.pdf'
CURVE_PATH = PROJECT_DIR / 'model' / 'training_curves.png'


def clean_inline_md(text: str) -> str:
    text = text.replace('**', '')
    text = text.replace('`', '')
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)
    return text.strip()


def parse_markdown(md_text: str):
    lines = md_text.splitlines()
    blocks = []
    i = 0
    in_list = False

    while i < len(lines):
        raw = lines[i].rstrip('\n')
        line = raw.strip()

        if line == '':
            blocks.append(('spacer', ''))
            in_list = False
            i += 1
            continue

        if line == '\\newpage':
            blocks.append(('pagebreak', ''))
            in_list = False
            i += 1
            continue

        if line.startswith('!['):
            blocks.append(('image', line))
            in_list = False
            i += 1
            continue

        if line.startswith('# '):
            blocks.append(('h1', clean_inline_md(line[2:])))
            in_list = False
            i += 1
            continue

        if line.startswith('## '):
            blocks.append(('h2', clean_inline_md(line[3:])))
            in_list = False
            i += 1
            continue

        if line.startswith('### '):
            blocks.append(('h3', clean_inline_md(line[4:])))
            in_list = False
            i += 1
            continue

        if line.startswith('- '):
            blocks.append(('li', clean_inline_md(line[2:])))
            in_list = True
            i += 1
            continue

        if re.match(r'^\d+\.\s+', line):
            blocks.append(('li', clean_inline_md(re.sub(r'^\d+\.\s+', '', line))))
            in_list = True
            i += 1
            continue

        blocks.append(('p', clean_inline_md(line)))
        in_list = False
        i += 1

    return blocks


def build_pdf():
    md_text = MD_PATH.read_text(encoding='utf-8')
    blocks = parse_markdown(md_text)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor('#0f2747'),
        alignment=1,
        spaceAfter=20,
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2f4f6f'),
        alignment=1,
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#12355b'),
        spaceBefore=10,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1d4e89'),
        spaceBefore=8,
        spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2a5d8f'),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=4,
    )

    story = []

    # Professional title page
    story.append(Spacer(1, 4.2 * cm))
    story.append(Paragraph('Dog Breed Identification Using Transfer Learning', title_style))
    story.append(Paragraph('Internship Mini Project', subtitle_style))
    story.append(Spacer(1, 1.6 * cm))
    story.append(Paragraph('Author: Mokshagnaram', subtitle_style))
    story.append(Paragraph(f'Date: {date(2026, 2, 19).strftime("%B %d, %Y")}', subtitle_style))
    story.append(Spacer(1, 8 * cm))
    story.append(Paragraph('Technical Report', ParagraphStyle(
        'FooterTitle', parent=subtitle_style, fontSize=12, textColor=colors.HexColor('#4f6f90')
    )))
    story.append(PageBreak())

    skip_initial_title_block = True
    for block_type, content in blocks:
        if skip_initial_title_block:
            if block_type in {'h1', 'h2'} and content in {
                'Dog Breed Identification Using Transfer Learning',
                'Internship Mini Project',
            }:
                continue
            if block_type == 'p' and (content.startswith('Author:') or content.startswith('Date:')):
                continue
            if block_type == 'pagebreak':
                skip_initial_title_block = False
                continue
            skip_initial_title_block = False

        if block_type == 'h1':
            story.append(Paragraph(content, h1_style))
        elif block_type == 'h2':
            story.append(Paragraph(content, h2_style))
        elif block_type == 'h3':
            story.append(Paragraph(content, h3_style))
        elif block_type == 'p':
            story.append(Paragraph(content, body_style))
        elif block_type == 'li':
            story.append(Paragraph(f'• {content}', bullet_style))
        elif block_type == 'spacer':
            story.append(Spacer(1, 0.12 * cm))
        elif block_type == 'pagebreak':
            story.append(PageBreak())
        elif block_type == 'image':
            if CURVE_PATH.exists():
                img = Image(str(CURVE_PATH))
                img._restrictSize(16.5 * cm, 9.5 * cm)
                story.append(Spacer(1, 0.2 * cm))
                story.append(img)
                story.append(Spacer(1, 0.25 * cm))
            else:
                story.append(Paragraph('Training curve image not found.', body_style))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#667085'))
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.2 * cm, f'Page {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title='Dog Breed Identification Using Transfer Learning',
        author='Mokshagnaram',
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == '__main__':
    build_pdf()
    print(f'Created: {PDF_PATH}')
