from __future__ import annotations

from math import cos, radians
from typing import Any


def _distance_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * cos(radians((float(a['latitude']) + float(b['latitude'])) / 2))
    return (((float(a['latitude']) - float(b['latitude'])) * lat_scale) ** 2 + ((float(a['longitude']) - float(b['longitude'])) * lon_scale) ** 2) ** 0.5


def build_segments(events: list[dict[str, Any]], route: dict[str, Any], segment_m: float = 100.0) -> list[dict[str, Any]]:
    points = route.get('points', [])
    if not points:
        return []
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [points[0]]
    length = 0.0
    segment_no = 1
    for point in points[1:]:
        length += _distance_m(current[-1], point)
        current.append(point)
        if length >= segment_m:
            segments.append(_segment(segment_no, current, events))
            segment_no += 1
            current = [point]
            length = 0.0
    if len(current) > 1:
        segments.append(_segment(segment_no, current, events))
    return segments


def _segment(number: int, points: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    latitudes = [float(p['latitude']) for p in points]
    longitudes = [float(p['longitude']) for p in points]
    center = {'latitude': sum(latitudes) / len(latitudes), 'longitude': sum(longitudes) / len(longitudes)}
    nearby = []
    for event in events:
        if event.get('latitude') is None or event.get('longitude') is None:
            continue
        distance = min(_distance_m(center, p) for p in points)
        event_distance = _distance_m(center, {'latitude': event['latitude'], 'longitude': event['longitude']})
        if event_distance <= 75:
            nearby.append((event, event_distance))
    potholes = [e for e, _ in nearby]
    high = sum(e.get('severity') == 'high' for e in potholes)
    medium = sum(e.get('severity') == 'medium' for e in potholes)
    low = sum(e.get('severity') == 'low' for e in potholes)
    score = max(0.0, 100.0 - high * 15.0 - medium * 8.0 - low * 3.0)
    if not potholes:
        priority = 'Monitor'
    elif high >= 2 or score < 45:
        priority = 'Urgent'
    elif high or medium >= 2 or score < 70:
        priority = 'High'
    else:
        priority = 'Routine'
    return {'segmentId': f'RS-{number:03d}', 'center': center, 'potholes': len(potholes), 'high': high, 'medium': medium, 'low': low, 'conditionScore': round(score, 1), 'priority': priority, 'eventIds': [e['event_id'] for e in potholes]}


def repair_queue(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {'Urgent': 0, 'High': 1, 'Routine': 2, 'Monitor': 3}
    return sorted(segments, key=lambda s: (order.get(s['priority'], 9), -s['potholes'], s['conditionScore']))
