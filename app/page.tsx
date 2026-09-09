'use client'

import { useEffect, useRef, useState } from 'react'

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [cameraOn, setCameraOn] = useState(false)
  const [inspecting, setInspecting] = useState(false)
  const [gps, setGps] = useState<'waiting' | 'ready' | 'unavailable'>('waiting')
  const [coordinates, setCoordinates] = useState<string>('Waiting for permission')

  useEffect(() => {
    if (!navigator.geolocation) {
      setGps('unavailable')
      setCoordinates('Geolocation is not supported')
      return
    }

    navigator.geolocation.getCurrentPosition(
      position => {
        setGps('ready')
        setCoordinates(`${position.coords.latitude.toFixed(5)}, ${position.coords.longitude.toFixed(5)}`)
      },
      () => {
        setGps('unavailable')
        setCoordinates('Location unavailable')
      },
      { enableHighAccuracy: true, timeout: 10000 }
    )
  }, [])

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraOn(true)
    } catch {
      setCameraOn(false)
      alert('Camera permission was not granted or no camera is available.')
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    setCameraOn(false)
    setInspecting(false)
  }

  function toggleInspection() {
    if (!cameraOn) {
      void startCamera()
      setInspecting(true)
      return
    }
    setInspecting(value => !value)
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">⌁</div>
          <div>
            <div className="brand-name">RoadScan AI</div>
            <div className="brand-sub">Road defect detection & inspection</div>
          </div>
        </div>
        <div className="status">
          <span className={`dot ${inspecting ? 'live' : ''}`} />
          {inspecting ? 'LIVE INSPECTION' : 'STANDBY'}
        </div>
      </header>

      <section className="shell">
        <div className="hero">
          <div>
            <h1>Live Road Inspection</h1>
            <p className="muted">Camera evidence first. AI detections, locations and reports are built from real inspection data.</p>
          </div>
          <div className="actions">
            <button className="btn" onClick={() => alert('Report generation will be connected after the detection pipeline is added.')}>View Reports</button>
            <button className="btn primary" onClick={toggleInspection}>{inspecting ? 'Pause Inspection' : 'Start Inspection'}</button>
          </div>
        </div>

        <div className="grid">
          <section className="card">
            <div className="camera-head">
              <div className="camera-title"><span>▣</span> Camera Feed</div>
              <div className="live-pill">{inspecting ? '● LIVE' : 'READY'}</div>
            </div>
            <div className="video-wrap">
              <video ref={videoRef} className="video" muted playsInline />
              {!cameraOn && (
                <div className="video-placeholder">
                  <div>
                    <strong>Camera feed is offline</strong>
                    <span>Start an inspection to activate the camera</span>
                  </div>
                </div>
              )}
              <div className="overlay">
                <div className="chip">Inference: {inspecting ? 'ACTIVE' : 'IDLE'}</div>
                <div className="chip">FPS: —</div>
              </div>
            </div>
            <div style={{padding:'14px 16px', display:'flex', justifyContent:'space-between', alignItems:'center', borderTop:'1px solid #1b2638'}}>
              <span className="muted">Real camera frames will feed the detector in the next milestone.</span>
              {cameraOn && <button className="btn danger" onClick={stopCamera}>Stop Camera</button>}
            </div>
          </section>

          <aside className="side">
            <section className="card panel">
              <div className="panel-title">Location</div>
              <div className="gps">
                <div>
                  <div style={{fontWeight:700}}>GPS {gps === 'ready' ? 'Ready' : gps === 'unavailable' ? 'Unavailable' : 'Waiting'}</div>
                  <div className="gps-note">{coordinates}</div>
                </div>
                <div className="ready">{gps === 'ready' ? '●' : '○'}</div>
              </div>
            </section>

            <section className="card panel">
              <div className="panel-title">Live Detection Summary</div>
              <div className="count">0</div>
              <div className="count-label">Unique potholes detected</div>
              <div className="severity">
                <div className="sev high"><b>0</b><span>HIGH</span></div>
                <div className="sev medium"><b>0</b><span>MEDIUM</span></div>
                <div className="sev low"><b>0</b><span>LOW</span></div>
              </div>
            </section>

            <section className="card panel">
              <div className="panel-title">Inspection telemetry</div>
              <div className="metric-row"><span>Frames processed</span><b>0</b></div>
              <div className="metric-row"><span>Unique detections</span><b>0</b></div>
              <div className="metric-row"><span>Evidence frames</span><b>0</b></div>
              <div className="metric-row"><span>GPS status</span><b>{gps === 'ready' ? 'Ready' : 'Unavailable'}</b></div>
            </section>
          </aside>
        </div>

        <div className="bottom">
          <section className="card">
            <div className="panel" style={{paddingBottom:10}}>
              <div className="panel-title">Survey Route</div>
            </div>
            <div className="route">
              <div className="map-label">Map preview — live defect markers arrive with the detection pipeline</div>
              <span className="map-dot red dot1" /><span className="map-dot orange dot2" /><span className="map-dot red dot3" /><span className="map-dot green dot4" />
            </div>
          </section>
          <section className="card panel">
            <div className="panel-title">Evidence Integrity</div>
            <div style={{fontSize:22, fontWeight:800, marginBottom:8}}>No fabricated detections</div>
            <p className="muted">RoadScan will create a pothole record only from a processed camera frame. Each record will retain its evidence frame, timestamp and GPS state.</p>
          </section>
        </div>
      </section>
    </main>
  )
}
