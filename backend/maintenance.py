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
    potholes = []
    for event in events:
        if event.get('latitude') is None or event.get('longitude') is None:
            continue
        if _distance_m(center, {'latitude': event['latitude'], 'longitude': event['longitude']}) <= 75:
            potholes.append(event)
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
    return {
        'segmentId': f'RS-{number:03d}', 'center': center, 'potholes': len(potholes),
        'high': high, 'medium': medium, 'low': low, 'conditionScore': round(score, 1),
        'priority': priority, 'eventIds': [e['event_id'] for e in potholes],
        'events': [{'eventId': e['event_id'], 'latitude': e['latitude'], 'longitude': e['longitude'],
                    'confidence': e.get('confidence', 0), 'confidenceBand': e.get('severity', 'low'),
                    'evidence': e.get('evidence', ''), 'timestamp': e.get('timestamp', '')} for e in potholes],
    }


def repair_queue(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {'Urgent': 0, 'High': 1, 'Routine': 2, 'Monitor': 3}
    return sorted(segments, key=lambda s: (order.get(s['priority'], 9), -s['potholes'], s['conditionScore']))


def find_recurring_potholes(inspections: list[dict[str, Any]], radius_m: float = 30.0) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for inspection in inspections:
        inspection_id = inspection.get('id', 'live')
        for event in inspection.get('events', []):
            if event.get('latitude') is None or event.get('longitude') is None:
                continue
            point = {'latitude': event['latitude'], 'longitude': event['longitude']}
            match = next((g for g in groups if inspection_id not in g['inspectionIds'] and _distance_m(g['center'], point) <= radius_m), None)
            if match is None:
                groups.append({'center': point, 'inspectionIds': [inspection_id], 'events': [event]})
            else:
                match['events'].append(event)
                match['inspectionIds'].append(inspection_id)
                n = len(match['events'])
                match['center'] = {'latitude': sum(float(e['latitude']) for e in match['events']) / n,
                                   'longitude': sum(float(e['longitude']) for e in match['events']) / n}
    recurring = []
    for index, group in enumerate(groups, 1):
        inspection_ids = list(dict.fromkeys(group['inspectionIds']))
        if len(inspection_ids) < 2:
            continue
        events = group['events']
        confidences = [float(e.get('confidence', 0)) for e in events]
        high = sum(e.get('severity') == 'high' for e in events)
        recurring.append({
            'recurringId': f'RP-{index:03d}', 'center': group['center'],
            'inspectionCount': len(inspection_ids), 'detectionCount': len(events),
            'priority': 'Urgent' if high >= 2 or len(inspection_ids) >= 3 else 'High',
            'averageConfidence': round(sum(confidences) / len(confidences), 3) if confidences else 0,
            'inspectionIds': inspection_ids,
            'eventIds': [e.get('event_id') or e.get('eventId') for e in events],
        })
    return sorted(recurring, key=lambda x: (-x['inspectionCount'], -x['detectionCount']))
