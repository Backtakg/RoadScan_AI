from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _fmt_timestamp(value: str | None) -> str:
    if not value:
        return "Unavailable"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _gps(event: dict[str, Any]) -> str:
    lat, lon = event.get("latitude"), event.get("longitude")
    if lat is None or lon is None:
        return "Unavailable"
    return f"{float(lat):.6f}, {float(lon):.6f}"


def _safe_text(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_inspection_pdf(events: list[dict[str, Any]], route: dict[str, Any], evidence_dir: str) -> bytes:
    """Build a self-contained inspection report from actual store data."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="RoadScan AI Inspection Report",
        author="RoadScan AI",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, leading=26, textColor=colors.HexColor("#0f172a"), alignment=TA_LEFT, spaceAfter=5 * mm)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#64748b"), spaceAfter=7 * mm)
    section = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#0f172a"), spaceBefore=4 * mm, spaceAfter=3 * mm)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#475569"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#334155"))
    centered = ParagraphStyle("Centered", parent=small, alignment=TA_CENTER)

    total = len(events)
    mapped = sum(e.get("latitude") is not None and e.get("longitude") is not None for e in events)
    high = sum(e.get("severity") == "high" for e in events)
    medium = sum(e.get("severity") == "medium" for e in events)
    low = sum(e.get("severity") == "low" for e in events)
    distance_m = float(route.get("distanceMeters", 0) or 0)
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    story: list[Any] = []
    story.append(Paragraph("RoadScan AI", title))
    story.append(Paragraph("Road inspection evidence report", subtitle))
    story.append(Paragraph("Inspection summary", section))
    summary_data = [
        ["Unique potholes", str(total), "Mapped detections", f"{mapped}/{total}"],
        ["Route distance", f"{distance_m / 1000:.3f} km", "Route points", str(route.get("pointCount", 0))],
        ["High confidence", str(high), "Medium / Low", f"{medium} / {low}"],
        ["Report generated", generated, "Evidence policy", "One frame per unique tracked event"],
    ]
    table = Table(summary_data, colWidths=[35 * mm, 48 * mm, 38 * mm, 53 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Data integrity note: this report contains only detections, GPS coordinates, route points, timestamps, and evidence frames produced by the RoadScan AI inspection session. Missing GPS is shown as unavailable rather than estimated.", body))

    if route.get("points"):
        story.append(Paragraph("Inspection route", section))
        points = route["points"]
        route_rows = [["#", "Latitude", "Longitude", "Timestamp"]]
        for i, point in enumerate(points, 1):
            route_rows.append([str(i), f"{float(point['latitude']):.6f}", f"{float(point['longitude']):.6f}", _fmt_timestamp(point.get("timestamp"))])
        route_table = Table(route_rows, repeatRows=1, colWidths=[12 * mm, 40 * mm, 40 * mm, 78 * mm])
        route_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(route_table)

    story.append(PageBreak())
    story.append(Paragraph("Detected potholes & evidence", section))
    if not events:
        story.append(Paragraph("No pothole events were recorded in this inspection session.", body))
    else:
        for event in events:
            event_id = _safe_text(event.get("event_id", "Unknown"))
            evidence = event.get("evidence", "")
            evidence_path = evidence_dir if not evidence else os.path.join(evidence_dir, os.path.basename(evidence))
            meta = [
                ["Event", event_id, "Track", str(event.get("track_id", "—"))],
                ["AI confidence", f"{float(event.get('confidence', 0)) * 100:.1f}%", "Confidence band", _safe_text(event.get("severity", "unknown"))],
                ["GPS", _gps(event), "Detected", _fmt_timestamp(event.get("timestamp"))],
            ]
            meta_table = Table(meta, colWidths=[30 * mm, 62 * mm, 35 * mm, 43 * mm])
            meta_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            flow: list[Any] = [meta_table, Spacer(1, 2 * mm)]
            if evidence and os.path.isfile(evidence_path):
                try:
                    img = Image(evidence_path)
                    img._restrictSize(170 * mm, 72 * mm)
                    flow.extend([img, Spacer(1, 2 * mm)])
                except Exception:
                    flow.append(Paragraph("Evidence image could not be embedded.", small))
            else:
                flow.append(Paragraph("Evidence image unavailable for this event.", small))
            story.append(KeepTogether(flow))
            story.append(Spacer(1, 5 * mm))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("AI confidence bands are detection-confidence groupings and are not an engineering assessment of pothole severity, depth, structural risk, or repair priority.", small))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(16 * mm, 11 * mm, 194 * mm, 11 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 7 * mm, "RoadScan AI · Evidence-based road inspection")
        canvas.drawRightString(194 * mm, 7 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
