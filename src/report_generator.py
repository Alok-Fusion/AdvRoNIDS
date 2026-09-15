"""
AdvRoNIDS PDF Report Generation Module
Generates publication-quality, attack-specific PDF threat reports and session incident audit reports
using ReportLab Platypus.
"""

import io
import time
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


def get_custom_styles():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569')
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=5,
        spaceBefore=10
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4,
        spaceBefore=6
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#334155')
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'callout': callout_style
    }


def build_attack_threat_report_pdf(attack_data: Dict[str, Any]) -> bytes:
    """
    Builds an Attack-Specific Threat Forensics & Mitigation Report PDF based on actual findings.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    st = get_custom_styles()
    story = []
    
    attack_name = attack_data.get("attack_name", "Adversarial Intrusion Attack")
    base_cat = attack_data.get("base_category", "Unknown Intrusion")
    actor = attack_data.get("threat_actor", "Unknown Threat Actor")
    mitre = attack_data.get("mitre_technique", "T1498 - Network Exploitation")
    target = attack_data.get("target_service", "Edge Gateway")
    eps = attack_data.get("epsilon", 0.12)
    steps = attack_data.get("num_steps", 7)
    flow_id = attack_data.get("flow_id", "flow_simulated")
    
    pred_a = attack_data.get("model_a_pred", "Benign")
    conf_a = float(attack_data.get("model_a_conf", 0.0)) * 100.0
    evaded_a = attack_data.get("model_a_evaded", True)
    
    pred_b = attack_data.get("model_b_pred", base_cat)
    conf_b = float(attack_data.get("model_b_conf", 0.0)) * 100.0
    
    forensics = attack_data.get("forensics") or {}
    exec_summary = forensics.get("executive_summary", attack_data.get("scenario_brief", "Adversarial evasion attempt against standard intrusion detection models."))
    root_cause = forensics.get("root_cause_analysis", "Gradient perturbation shifted continuous timing features into benign probability simplex.")
    mitigation = forensics.get("mitigation_playbook", "1. Deploy AdvRoNIDS invariant filters.\n2. Isolate compromised socket ports.")
    
    top_features = attack_data.get("top_features", [])

    # 1. Document Header Banner
    story.append(Paragraph(f"AdvRoNIDS Threat Assessment: {attack_name}", st['title']))
    story.append(Spacer(1, 3))
    date_str = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Incident Classification: <b>{base_cat}</b> | Generated on {date_str} | AdvRoNIDS AI Defense", st['subtitle']))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e40af'), spaceAfter=10))
    
    # 2. Threat Profile Matrix Table
    profile_data = [
        [
            Paragraph(f"<b>Threat Signature:</b> {attack_name}", st['body']),
            Paragraph(f"<b>Attributed Actor:</b> {actor}", st['body'])
        ],
        [
            Paragraph(f"<b>MITRE ATT&CK:</b> <font color='#1e40af'><b>{mitre}</b></font>", st['body']),
            Paragraph(f"<b>Targeted Service:</b> {target}", st['body'])
        ],
        [
            Paragraph(f"<b>Perturbation Budget:</b> &epsilon; = {eps:.2f} ({steps} PGD Steps)", st['body']),
            Paragraph(f"<b>Constraint Space:</b> Realistic Domain-Preserving (&Pi;<sub>S</sub>)", st['body'])
        ]
    ]
    t_profile = Table(profile_data, colWidths=[270, 270])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 10))
    
    # 3. Dual Model AI Decision Comparison
    story.append(Paragraph("1. Dual Model Detection Verdicts", st['h1']))
    verdict_data = [
        [
            Paragraph("<b>Detection Engine</b>", st['body']),
            Paragraph("<b>Classification Verdict</b>", st['body']),
            Paragraph("<b>Confidence Score</b>", st['body']),
            Paragraph("<b>Defensive Status</b>", st['body'])
        ],
        [
            Paragraph("<b>Standard DNN (Model A)</b>", st['body']),
            Paragraph(f"<font color='#dc2626'><b>{pred_a}</b></font>", st['body']),
            Paragraph(f"{conf_a:.1f}%", st['body']),
            Paragraph("<font color='#dc2626'><b>COMPROMISED (EVADED)</b></font>" if evaded_a else "NORMAL", st['body'])
        ],
        [
            Paragraph("<b>AdvRoNIDS Robust (Model B)</b>", st['body']),
            Paragraph(f"<font color='#059669'><b>{pred_b}</b></font>", st['body']),
            Paragraph(f"{conf_b:.1f}%", st['body']),
            Paragraph("<font color='#059669'><b>DEFENSE INTACT (IDENTIFIED)</b></font>", st['body'])
        ]
    ]
    t_verdict = Table(verdict_data, colWidths=[150, 130, 110, 150])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#fef2f2') if evaded_a else colors.HexColor('#ffffff')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#f0fdf4')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 10))

    # 4. Executive Forensics & Root Cause Analysis
    story.append(Paragraph("2. Threat Intelligence & Forensics Analysis", st['h1']))
    story.append(Paragraph(f"<b>Executive Assessment:</b> {exec_summary}", st['body']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Exploitation Vector:</b> {root_cause}", st['body']))
    story.append(Spacer(1, 10))

    # 5. Top Perturbed Network Features
    if top_features:
        story.append(Paragraph("3. Adversarial Feature Shift Telemetry", st['h1']))
        feat_rows = [
            [
                Paragraph("<b>Feature Name</b>", st['body']),
                Paragraph("<b>Domain Role</b>", st['body']),
                Paragraph("<b>Model A Delta (&Delta;<sub>A</sub>)</b>", st['body']),
                Paragraph("<b>Model B Delta (&Delta;<sub>B</sub>)</b>", st['body'])
            ]
        ]
        for f in top_features[:5]:
            feat_rows.append([
                Paragraph(f"<b>{f.get('feature_name', '')}</b>", st['body']),
                Paragraph("Structural Invariant" if f.get("is_frozen") else "Manipulable", st['body']),
                Paragraph(f"<font color='#dc2626'>{f.get('delta_a', 0.0):+.4f}</font>", st['body']),
                Paragraph(f"<font color='#059669'>{f.get('delta_b', 0.0):+.4f}</font>", st['body'])
            ])
        t_feat = Table(feat_rows, colWidths=[180, 120, 120, 120])
        t_feat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_feat)
        story.append(Spacer(1, 10))

    # 6. Actionable Mitigation Playbook
    story.append(Paragraph("4. Recommended SOC Mitigation Playbook", st['h1']))
    if isinstance(mitigation, list):
        mit_lines = [str(m) for m in mitigation]
    elif isinstance(mitigation, str):
        mit_lines = mitigation.split('\n')
    else:
        mit_lines = [str(mitigation)]

    for ml in mit_lines:
        cleaned_line = str(ml).strip().lstrip('1234567890. -•')
        if cleaned_line:
            story.append(Paragraph(f"• {cleaned_line}", st['body']))
            story.append(Spacer(1, 2))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_session_report_pdf(session_data: Dict[str, Any]) -> bytes:
    """
    Builds a full session SOC Incident Report PDF summarizing all intercepted flows.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    st = get_custom_styles()
    story = []
    
    story.append(Paragraph("AdvRoNIDS — SOC Session Incident Audit Report", st['title']))
    story.append(Spacer(1, 3))
    date_str = time.strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Session History Audit | Generated on {date_str} | AdvRoNIDS AI Defense", st['subtitle']))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e40af'), spaceAfter=10))
    
    flows_count = session_data.get("total_flows", 0)
    b_caught = session_data.get("b_caught", 0)
    a_missed = session_data.get("a_missed", 0)
    def_rate = session_data.get("defense_rate", "100.0%")
    incidents = session_data.get("incidents", [])
    
    summary_data = [
        [
            Paragraph(f"<b>Total Flows Scanned:</b> {flows_count}", st['body']),
            Paragraph(f"<b>Model B Caught:</b> <font color='#059669'><b>{b_caught}</b></font>", st['body'])
        ],
        [
            Paragraph(f"<b>Model A Evaded:</b> <font color='#dc2626'><b>{a_missed}</b></font>", st['body']),
            Paragraph(f"<b>AdvRoNIDS Defense Rate:</b> <font color='#0284c7'><b>{def_rate}</b></font>", st['body'])
        ]
    ]
    t_summary = Table(summary_data, colWidths=[270, 270])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Incident Triage Audit Log", st['h1']))
    if not incidents:
        story.append(Paragraph("<i>No incidents triaged in this session yet.</i>", st['body']))
    else:
        table_rows = [
            [
                Paragraph("<b>Time</b>", st['body']),
                Paragraph("<b>Flow ID</b>", st['body']),
                Paragraph("<b>True Threat</b>", st['body']),
                Paragraph("<b>Model A</b>", st['body']),
                Paragraph("<b>Model B</b>", st['body']),
                Paragraph("<b>AI Briefing</b>", st['body'])
            ]
        ]
        for inc in incidents[-15:]:
            gt = inc.get("ground_truth_label", "Unknown")
            pa = inc.get("model_a_prediction", {})
            pb = inc.get("model_b_prediction", {})
            ev_a = pa.get("label") != gt
            table_rows.append([
                Paragraph(inc.get("timestamp", ""), st['body']),
                Paragraph(inc.get("flow_id", "")[:14], st['body']),
                Paragraph(f"<b>{gt}</b>", st['body']),
                Paragraph(f"<font color='{'#dc2626' if ev_a else '#334155'}'>{pa.get('label','')} ({(pa.get('confidence',0)*100):.0f}%)</font>", st['body']),
                Paragraph(f"<font color='#059669'><b>{pb.get('label','')} ({(pb.get('confidence',0)*100):.0f}%)</b></font>", st['body']),
                Paragraph(f"<font size='7'>{inc.get('incident_note', '')[:100]}...</font>", st['body'])
            ])
        t_inc = Table(table_rows, colWidths=[55, 75, 80, 95, 95, 140])
        t_inc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 5),
            ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_inc)
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
