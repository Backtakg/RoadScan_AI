'use client'

import dynamic from 'next/dynamic'
import { useEffect, useRef, useState } from 'react'

const InspectionMap = dynamic(() => import('../components/InspectionMap'), { ssr: false })

type Detection = { label: string; confidence: number; bbox: [number, number, number, number]; trackId: number | null }
type EventRecord = { event_id: string; track_id: number; timestamp: string; latitude: number | null; longitude: number | null; confidence: number; bbox: number[]; severity: 'high' | 'medium' | 'low'; evidence: string }
type Summary = { total: number; high: number; medium: number; low: number }
const API_URL = process.env.NEXT_PUBLIC_VISION_API_URL || 'http://localhost:8000'
const emptySummary: Summary = { total: 0, high: 0, medium: 0, low: 0 }

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null); const canvasRef = useRef<HTMLCanvasElement>(null); const streamRef = useRef<MediaStream | null>(null); const timerRef = useRef<ReturnType<typeof setInterval> | null>(null); const busyRef = useRef(false)
  const [cameraOn, setCameraOn] = useState(false); const [inspecting, setInspecting] = useState(false); const [gps, setGps] = useState<'waiting' | 'ready' | 'unavailable'>('waiting'); const [gpsPoint, setGpsPoint] = useState<[number, number] | null>(null); const [coordinates, setCoordinates] = useState('Waiting for permission'); const [frames, setFrames] = useState(0); const [inferenceMs, setInferenceMs] = useState<number | null>(null); const [detections, setDetections] = useState<Detection[]>([]); const [events, setEvents] = useState<EventRecord[]>([]); const [summary, setSummary] = useState<Summary>(emptySummary); const [apiStatus, setApiStatus] = useState<'offline' | 'online'>('offline')

  useEffect(() => {
    if (!navigator.geolocation) return setGps('unavailable')
    const id = navigator.geolocation.watchPosition(p => { setGps('ready'); setGpsPoint([p.coords.latitude, p.coords.longitude]); setCoordinates(`${p.coords.latitude.toFixed(5)}, ${p.coords.longitude.toFixed(5)}`) }, () => { setGps('unavailable'); setCoordinates('Location unavailable') }, { enableHighAccuracy: true, maximumAge: 2000, timeout: 10000 })
    return () => navigator.geolocation.clearWatch(id)
  }, [])

  async function startCamera() { try { const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false }); streamRef.current = stream; if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }; setCameraOn(true); return true } catch { setCameraOn(false); alert('Camera permission was not granted or no camera is available.'); return false } }
  async function resetInspection() { try { await fetch(`${API_URL}/reset`, { method: 'POST' }) } catch {} setEvents([]); setSummary(emptySummary); setDetections([]); setFrames(0) }
  function stopCamera() { if (timerRef.current) clearInterval(timerRef.current); streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null; timerRef.current = null; setCameraOn(false); setInspecting(false) }

  async function processFrame() {
    if (busyRef.current) return
    const video = videoRef.current, canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2 || !inspecting || !video.videoWidth) return
    busyRef.current = true; const width = Math.min(video.videoWidth, 960), height = Math.round(video.videoHeight / video.videoWidth * width); canvas.width = width; canvas.height = height; canvas.getContext('2d')?.drawImage(video, 0, 0, width, height)
    try {
      const r = await fetch(`${API_URL}/detect`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.72), latitude: gpsPoint?.[0] ?? null, longitude: gpsPoint?.[1] ?? null, timestamp: new Date().toISOString() }) })
      if (!r.ok) throw new Error('detector unavailable'); const data = await r.json(); setDetections(data.detections || []); setEvents(data.events || []); setSummary(data.summary || emptySummary); setInferenceMs(data.inferenceMs ?? null); setApiStatus('online'); setFrames(v => v + 1)
    } catch { setApiStatus('offline') } finally { busyRef.current = false }
  }
  async function toggleInspection() { if (inspecting) { setInspecting(false); if (timerRef.current) clearInterval(timerRef.current); timerRef.current = null; return }; const ready = cameraOn || await startCamera(); if (ready) { await resetInspection(); setInspecting(true) } }
  useEffect(() => { if (!inspecting) return; timerRef.current = setInterval(() => void processFrame(), 500); return () => { if (timerRef.current) clearInterval(timerRef.current) } }, [inspecting, gpsPoint])

  return <main className="app">
    <header className="topbar"><div className="brand"><div className="brand-mark">⌁</div><div><div className="brand-name">RoadScan AI</div><div className="brand-sub">Road defect detection & inspection</div></div></div><div className="status"><span className={`dot ${inspecting ? 'live' : ''}`} />{inspecting ? 'LIVE INSPECTION' : 'STANDBY'}</div></header>
    <section className="shell">
      <div className="hero"><div><h1>Live Road Inspection</h1><p className="muted">Camera → AI tracking → GPS evidence → inspection map.</p></div><div className="actions"><button className="btn" onClick={() => document.getElementById('map-section')?.scrollIntoView({ behavior: 'smooth' })}>View Map</button><button className="btn primary" onClick={() => void toggleInspection()}>{inspecting ? 'Pause Inspection' : 'Start Inspection'}</button></div></div>
      <div className="grid"><section className="card"><div className="camera-head"><div className="camera-title"><span>▣</span> Camera Feed</div><div className="live-pill">{inspecting ? '● LIVE' : cameraOn ? 'READY' : 'OFFLINE'}</div></div><div className="video-wrap"><video ref={videoRef} className="video" muted playsInline /><canvas ref={canvasRef} hidden />{detections.map((d, i) => { const v = videoRef.current; if (!v?.videoWidth) return null; const [x1,y1,x2,y2]=d.bbox; return <div key={`${d.trackId ?? 'untracked'}-${i}`} className="bbox" style={{left:`${x1/v.videoWidth*100}%`,top:`${y1/v.videoHeight*100}%`,width:`${(x2-x1)/v.videoWidth*100}%`,height:`${(y2-y1)/v.videoHeight*100}%`}}><span>{d.label} {d.trackId != null ? `#${d.trackId} ` : ''}{(d.confidence*100).toFixed(0)}%</span></div> })}{!cameraOn && <div className="video-placeholder"><div><strong>Camera feed is offline</strong><span>Start an inspection to activate the camera</span></div></div>}<div className="overlay"><div className="chip">Inference: {inspecting ? 'ACTIVE' : 'IDLE'}</div><div className="chip">API: {apiStatus.toUpperCase()}</div><div className="chip">Latency: {inferenceMs == null ? '—' : `${inferenceMs} ms`}</div></div></div><div style={{padding:'14px 16px',display:'flex',justifyContent:'space-between',alignItems:'center',borderTop:'1px solid #1b2638'}}><span className="muted">Unique potholes are counted from persistent tracker events.</span>{cameraOn && <button className="btn danger" onClick={stopCamera}>Stop Camera</button>}</div></section>
      <aside className="side"><section className="card panel"><div className="panel-title">Location</div><div className="gps"><div><div style={{fontWeight:700}}>GPS {gps === 'ready' ? 'Ready' : gps === 'unavailable' ? 'Unavailable' : 'Waiting'}</div><div className="gps-note">{coordinates}</div></div><div className="ready">{gps === 'ready' ? '●' : '○'}</div></div></section><section className="card panel"><div className="panel-title">Unique Potholes</div><div className="count">{summary.total}</div><div className="count-label">Tracked inspection events</div><div className="severity"><div className="sev high"><b>{summary.high}</b><span>HIGH</span></div><div className="sev medium"><b>{summary.medium}</b><span>MEDIUM</span></div><div className="sev low"><b>{summary.low}</b><span>LOW</span></div></div></section><section className="card panel"><div className="panel-title">Inspection telemetry</div><div className="metric-row"><span>Frames processed</span><b>{frames}</b></div><div className="metric-row"><span>Current detections</span><b>{detections.length}</b></div><div className="metric-row"><span>Unique potholes</span><b>{summary.total}</b></div><div className="metric-row"><span>GPS mapped</span><b>{events.filter(e => e.latitude != null && e.longitude != null).length}</b></div><div className="metric-row"><span>Detector API</span><b>{apiStatus === 'online' ? 'Online' : 'Offline'}</b></div></section></aside></div>
      <div className="bottom"><section className="card" id="map-section"><div className="panel" style={{paddingBottom:10}}><div className="panel-title">Inspection Route & Pothole Map</div></div><InspectionMap events={events} gps={gpsPoint} /></section><section className="card panel"><div className="panel-title">Tracked Events</div><div className="event-list">{events.length === 0 ? <div className="empty-events">No pothole events yet. Start inspection and detections with valid GPS will appear here.</div> : events.slice().reverse().slice(0,8).map(e => <div className="event-row" key={e.event_id}><div><b>{e.event_id}</b><span>Track #{e.track_id} · {e.latitude != null ? `${e.latitude.toFixed(5)}, ${e.longitude?.toFixed(5)}` : 'GPS unavailable'}</span></div><strong className={`event-${e.severity}`}>{(e.confidence * 100).toFixed(0)}%</strong></div>)}</div><button className="btn danger reset-btn" onClick={() => void resetInspection()}>Reset Inspection Data</button></section></div>
    </section>
  </main>
}
