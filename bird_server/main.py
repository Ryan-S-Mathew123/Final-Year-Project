import asyncio, threading, time
from collections import deque
from pathlib import Path
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from classifier import BirdClassifier
from localization import estimate_delay,multilateration

FS=44100; MAX_SAMPLES=FS*10; WINDOW=FS*2
lock=threading.Lock()
phones={}
buffers={}
connected=set()
last=0.0
classifier=BirdClassifier()
result={"status":"Ready — add phones or connect phone apps","species":"Waiting","confidence":0.0,"position":None,"error_radius":None}

class PhoneCreate(BaseModel):
    phone_id:str
    x:float=0
    y:float=0
class Position(BaseModel):
    x:float
    y:float

app=FastAPI(title="Bird Localization")

def clean_id(s):
    return "".join(c for c in s.strip() if c.isalnum() or c in "_-")[:40]

def add_phone(pid,x=0,y=0):
    pid=clean_id(pid)
    if not pid:raise ValueError("Phone ID must contain letters or numbers")
    with lock:
        if pid not in phones:
            phones[pid]=[float(x),float(y)]
            buffers[pid]=deque(maxlen=MAX_SAMPLES)
    return pid

@app.get("/")
async def home(): return FileResponse(Path(__file__).parent/"dashboard"/"index.html")

@app.get("/phones")
async def get_phones():
    with lock:return [{"id":i,"x":p[0],"y":p[1],"connected":i in connected} for i,p in sorted(phones.items())]

@app.post("/phones")
async def create_phone(data:PhoneCreate):
    try:pid=add_phone(data.phone_id,data.x,data.y)
    except ValueError as e:raise HTTPException(400,str(e))
    return {"id":pid,"x":data.x,"y":data.y}

@app.put("/phones/{pid}/position")
async def move_phone(pid:str,data:Position):
    if data.x<0 or data.y<0:raise HTTPException(400,"Coordinates must be non-negative")
    with lock:
        if pid not in phones:raise HTTPException(404,"Phone not found")
        phones[pid]=[data.x,data.y]
    return {"ok":True}

@app.delete("/phones/{pid}")
async def delete_phone(pid:str):
    with lock:
        if pid not in phones:raise HTTPException(404,"Phone not found")
        if pid in connected:raise HTTPException(409,"Disconnect the phone before deleting it")
        phones.pop(pid);buffers.pop(pid,None)
    return {"ok":True}

def process():
    global last,result
    now=time.time()
    with lock:
        if now-last<1:return
        last=now; ids=sorted(connected)
        if len(ids)<3:
            result={"status":f"Waiting for 3 connected phones ({len(ids)}/3)","species":"Waiting","confidence":0,"position":None,"error_radius":None};return
        if any(len(buffers[i])<WINDOW for i in ids):
            result={"status":"Phones connected — filling 2-second audio window","species":"Waiting","confidence":0,"position":None,"error_radius":None};return
        pos={i:list(phones[i]) for i in ids}; aud={i:np.asarray(list(buffers[i])[-WINDOW:],np.float32) for i in ids}
    ref=ids[0]; pred=classifier.predict(aud[ref],FS)
    if not pred["detected"]:
        with lock:result={"status":"No bird detected","species":pred["species"],"confidence":pred["confidence"],"position":None,"error_radius":None};return
    delays={ref:0.0}
    for i in ids[1:]:delays[i]=estimate_delay(aud[ref],aud[i],FS)
    try:loc=multilateration(pos,delays,ref)
    except Exception as e:
        with lock:result={"status":str(e),"species":pred["species"],"confidence":pred["confidence"],"position":None,"error_radius":None};return
    with lock:result={"status":"Bird localized (prototype)","species":pred["species"],"confidence":pred["confidence"],"position":loc["position"],"error_radius":max(.5,loc["error"]*2)}

@app.get("/result")
async def get_result():
    with lock:
        r=dict(result);r["connected"]=sorted(connected);r["connected_count"]=len(connected);return r

@app.websocket("/ws")
async def websocket(ws:WebSocket):
    await ws.accept();pid=None
    try:
        first=await ws.receive();txt=first.get("text","")
        if not txt.startswith("PHONE_ID:"):
            await ws.close(code=1008);return
        pid=add_phone(txt.split(":",1)[1])
        with lock:connected.add(pid)
        while True:
            m=await ws.receive();data=m.get("bytes")
            if data:
                a=np.frombuffer(data,dtype="<i2").astype(np.float32)/32768
                with lock:buffers[pid].extend(a.tolist())
                await asyncio.to_thread(process)
    except WebSocketDisconnect:pass
    finally:
        if pid:
            with lock:connected.discard(pid)
