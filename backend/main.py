from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timezone
from typing import Any
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO
from history_store import history
from inspection_store import store
from maintenance import build_segments, repair_queue
from report_generator import build_inspection_pdf
MODEL_PATH=os.getenv('ROADSCAN_MODEL','backend/models/best.pt'); TRACKER_PATH=os.getenv('ROADSCAN_TRACKER','backend/trackers/roadscan_bytetrack.yaml'); CONFIDENCE=float(os.getenv('ROADSCAN_CONFIDENCE','0.35')); EVIDENCE_DIR=os.getenv('ROADSCAN_EVIDENCE_DIR','backend/data/evidence'); os.makedirs(EVIDENCE_DIR,exist_ok=True)
app=FastAPI(title='RoadScan AI Vision API',version='0.9.0'); app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:3000','http://127.0.0.1:3000'],allow_credentials=True,allow_methods=['*'],allow_headers=['*']); app.mount('/evidence',StaticFiles(directory=EVIDENCE_DIR),name='evidence')
_model:YOLO|None=None; _inspection_started_at:str|None=None; _active_frames=0
class FrameRequest(BaseModel):
 image:str=Field(description='Base64 JPEG/PNG data, optionally prefixed with a data URL'); latitude:float|None=None; longitude:float|None=None; timestamp:str|None=None
def get_model()->YOLO:
 global _model
 if _model is None:
  if not os.path.exists(MODEL_PATH): raise HTTPException(status_code=503,detail=f'Pothole model not found at {MODEL_PATH}. Place a trained pothole YOLO weights file there or set ROADSCAN_MODEL.')
  _model=YOLO(MODEL_PATH)
 return _model
def decode_image(value:str)->np.ndarray:
 try:
  payload=value.split(',',1)[1] if ',' in value else value; image=cv2.imdecode(np.frombuffer(base64.b64decode(payload),dtype=np.uint8),cv2.IMREAD_COLOR)
  if image is None: raise ValueError('Unable to decode image')
  return image
 except Exception as exc: raise HTTPException(status_code=400,detail=f'Invalid image: {exc}') from exc
def save_evidence(image:np.ndarray,event_id:str)->str:
 filename=f'{event_id}.jpg'; path=os.path.join(EVIDENCE_DIR,filename)
 if not cv2.imwrite(path,image,[cv2.IMWRITE_JPEG_QUALITY,88]): raise RuntimeError(f'Unable to save evidence frame to {path}')
 return f'/evidence/{filename}'
def archive_current_inspection()->str|None:
 route=store.route_data(); events=store.list_events()
 if not events and not route['points']: return None
 now=datetime.now(timezone.utc).isoformat(); return history.save(_inspection_started_at or now,now,_active_frames,store.summary(),store.analytics(),events,route)
def maintenance_snapshot()->dict[str,Any]:
 segments=build_segments(store.list_events(),store.route_data()); return {'segments':segments,'repairQueue':repair_queue(segments),'segmentCount':len(segments)}
@app.get('/health')
def health()->dict[str,Any]: return {'status':'ok','model':MODEL_PATH,'model_exists':os.path.exists(MODEL_PATH),'model_loaded':_model is not None,'tracker':TRACKER_PATH,'tracker_exists':os.path.exists(TRACKER_PATH),'evidence_directory':EVIDENCE_DIR,'route_points':store.route_data()['pointCount'],'history_database':history.path,'active_frames':_active_frames,'inspection_active':_inspection_started_at is not None}
@app.post('/detect')
def detect(request:FrameRequest)->dict[str,Any]:
 global _inspection_started_at,_active_frames
 if _inspection_started_at is None: _inspection_started_at=request.timestamp or datetime.now(timezone.utc).isoformat()
 image=decode_image(request.image); started=time.perf_counter(); model=get_model(); _active_frames+=1; store.add_route_point(request.latitude,request.longitude,request.timestamp)
 results=model.track(image,conf=CONFIDENCE,tracker=TRACKER_PATH,persist=True,verbose=False); elapsed_ms=round((time.perf_counter()-started)*1000,1); detections=[]; new_events=[]; result=results[0]; names,boxes=result.names,result.boxes; track_ids=boxes.id.int().tolist() if boxes is not None and boxes.id is not None else []
 if boxes is not None:
  for index,box in enumerate(boxes):
   cls_id,confidence=int(box.cls.item()),float(box.conf.item()); bbox=[round(float(v),1) for v in box.xyxy[0].tolist()]; track_id=int(track_ids[index]) if index<len(track_ids) else None; detections.append({'classId':cls_id,'label':names.get(cls_id,str(cls_id)),'confidence':round(confidence,3),'bbox':bbox,'trackId':track_id})
   if track_id is not None:
    event,is_new=store.upsert(track_id,{'confidence':confidence,'bbox':bbox},request.latitude,request.longitude,request.timestamp)
    if is_new:
     try: store.set_evidence(event.event_id,save_evidence(image,event.event_id))
     except Exception as exc: raise HTTPException(status_code=500,detail=f'Evidence capture failed: {exc}') from exc
     new_events.append(event.to_dict())
 return {'detections':detections,'events':store.list_events(),'newEvents':new_events,'summary':store.summary(),'analytics':store.analytics(),'route':store.route_data(),'maintenance':maintenance_snapshot(),'inferenceMs':elapsed_ms,'timestamp':request.timestamp,'gps':{'latitude':request.latitude,'longitude':request.longitude},'source':'processed_camera_frame','model':MODEL_PATH,'tracker':TRACKER_PATH,'frames':_active_frames}
@app.get('/events')
def events()->dict[str,Any]: return {'events':store.list_events(),'summary':store.summary(),'analytics':store.analytics(),'route':store.route_data(),'maintenance':maintenance_snapshot(),'frames':_active_frames}
@app.get('/route')
def route()->dict[str,Any]: return store.route_data()
@app.get('/summary')
def summary()->dict[str,Any]: return store.summary()
@app.get('/analytics')
def analytics()->dict[str,Any]: return store.analytics()
@app.get('/maintenance')
def maintenance()->dict[str,Any]: return maintenance_snapshot()
@app.post('/history/start')
def history_start()->dict[str,Any]:
 global _inspection_started_at,_active_frames,_model
 if _inspection_started_at is not None and (store.list_events() or store.route_data()['points']): archive_current_inspection()
 store.reset(); _model=None; _active_frames=0; _inspection_started_at=datetime.now(timezone.utc).isoformat(); return {'status':'started','startedAt':_inspection_started_at}
@app.post('/history/complete')
def history_complete()->dict[str,Any]:
 global _inspection_started_at,_active_frames,_model
 inspection_id=archive_current_inspection()
 if inspection_id is None: raise HTTPException(status_code=400,detail='No inspection data to archive')
 saved=history.get(inspection_id); store.reset(); _model=None; _active_frames=0; _inspection_started_at=None; return {'status':'completed','id':inspection_id,'inspection':saved}
@app.get('/history')
def history_list(limit:int=50)->dict[str,Any]: return {'inspections':history.list(limit)}
@app.get('/history/trends')
def history_trends(limit:int=20)->dict[str,Any]: return {'trends':history.trends(limit)}
@app.get('/history/compare')
def history_compare(first:str,second:str)->dict[str,Any]:
 comparison=history.compare(first,second)
 if comparison is None: raise HTTPException(status_code=404,detail='One or both inspections were not found')
 return comparison
@app.get('/history/{inspection_id}/maintenance')
def history_maintenance(inspection_id:str)->dict[str,Any]:
 item=history.get(inspection_id)
 if item is None: raise HTTPException(status_code=404,detail='Inspection not found')
 segments=build_segments(item['events'],item['route']); return {'inspectionId':inspection_id,'segments':segments,'repairQueue':repair_queue(segments),'segmentCount':len(segments)}
@app.get('/history/{inspection_id}')
def history_detail(inspection_id:str)->dict[str,Any]:
 item=history.get(inspection_id)
 if item is None: raise HTTPException(status_code=404,detail='Inspection not found')
 return item
@app.delete('/history/{inspection_id}')
def history_delete(inspection_id:str)->dict[str,Any]:
 if not history.delete(inspection_id): raise HTTPException(status_code=404,detail='Inspection not found')
 return {'status':'ok','id':inspection_id}
@app.post('/history/save')
def history_save(frames:int=0)->dict[str,Any]:
 global _inspection_started_at
 if not store.list_events() and not store.route_data()['points']: raise HTTPException(status_code=400,detail='No inspection data to save')
 now=datetime.now(timezone.utc).isoformat(); started=_inspection_started_at or now; inspection_id=history.save(started,now,frames or _active_frames,store.summary(),store.analytics(),store.list_events(),store.route_data()); return {'status':'saved','id':inspection_id}
@app.get('/report.pdf',response_class=Response)
def report_pdf()->Response:
 try: pdf=build_inspection_pdf(store.list_events(),store.route_data(),EVIDENCE_DIR)
 except Exception as exc: raise HTTPException(status_code=500,detail=f'Report generation failed: {exc}') from exc
 filename=f'roadscan-inspection-{time.strftime("%Y%m%d-%H%M%S")}.pdf'; return Response(content=pdf,media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="{filename}"'})
@app.post('/reset')
def reset()->dict[str,Any]:
 global _model,_inspection_started_at,_active_frames
 store.reset(); _model=None; _inspection_started_at=None; _active_frames=0; return {'status':'ok','summary':store.summary(),'analytics':store.analytics(),'route':store.route_data()}
