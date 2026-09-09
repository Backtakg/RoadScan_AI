'use client'

import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from 'react-leaflet'
import type { LatLngExpression } from 'leaflet'
import { useEffect } from 'react'
import 'leaflet/dist/leaflet.css'

type Event = {
  event_id: string
  latitude: number | null
  longitude: number | null
  confidence: number
  severity: string
  timestamp: string
}
type RoutePoint = { latitude: number; longitude: number; timestamp: string }

function Recenter({ center }: { center: LatLngExpression | null }) {
  const map = useMap()
  useEffect(() => {
    if (center) map.setView(center, Math.max(map.getZoom(), 15), { animate: true })
  }, [center, map])
  return null
}

export default function InspectionMap({ events, gps, routePoints = [] }: { events: Event[]; gps: [number, number] | null; routePoints?: RoutePoint[] }) {
  const mapped = events.filter(e => Number.isFinite(e.latitude) && Number.isFinite(e.longitude))
  const center = gps || (routePoints.length ? [routePoints[routePoints.length - 1].latitude, routePoints[routePoints.length - 1].longitude] : null) || (mapped.length ? [mapped[mapped.length - 1].latitude as number, mapped[mapped.length - 1].longitude as number] : null)
  const route: LatLngExpression[] = routePoints.map(point => [point.latitude, point.longitude])

  if (!center) {
    return <div className="map-empty"><strong>Waiting for real GPS data</strong><span>The route appears only after the detector receives valid latitude/longitude samples.</span></div>
  }

  return <div className="map-canvas">
    <MapContainer center={center} zoom={15} scrollWheelZoom className="leaflet-map">
      <TileLayer attribution='&copy; OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <Recenter center={center} />
      {route.length > 1 && <Polyline positions={route} pathOptions={{ color: '#60a5fa', weight: 5, opacity: 0.9 }} />}
      {mapped.map(event => <CircleMarker key={event.event_id} center={[event.latitude as number, event.longitude as number]} radius={9} pathOptions={{ color: event.severity === 'high' ? '#ef4444' : event.severity === 'medium' ? '#f97316' : '#22c55e', fillOpacity: 0.9 }}>
        <Popup><strong>{event.event_id}</strong><br />Confidence: {(event.confidence * 100).toFixed(0)}%<br />AI confidence band: {event.severity}<br />{new Date(event.timestamp).toLocaleString()}</Popup>
      </CircleMarker>)}
      <CircleMarker center={center} radius={5} pathOptions={{ color: '#2563eb', fillColor: '#60a5fa', fillOpacity: 1 }}>
        <Popup>Current GPS position</Popup>
      </CircleMarker>
    </MapContainer>
    <div className="map-legend"><span><i className="legend-current" /> Current GPS</span><span><i className="legend-route" /> Inspection route</span><span><i className="legend-high" /> High confidence</span><span><i className="legend-medium" /> Medium confidence</span><span><i className="legend-low" /> Low confidence</span></div>
  </div>
}
