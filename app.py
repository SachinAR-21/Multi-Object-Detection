import os, json, base64, time, threading, io, multiprocessing as mp
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2, numpy as np
from PIL import Image
from bson import ObjectId
from datetime import datetime, timezone
from db import get_db, init_db

app = Flask(__name__)
app.secret_key = "nextvision_2024"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_IMG   = {"png","jpg","jpeg","bmp","webp"}
ALLOWED_VID   = {"mp4","avi","mov","mkv"}
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ── Load YOLO in background ───────────────────────────────────────────────────
model        = None
model_ready  = False
webcam_model = None

# ── Webcam: separate PROCESS for YOLO so it never steals CPU from Flask/video ─
_wcam_in  = None   # mp.Queue  frame  -> worker
_wcam_out = None   # mp.Queue  result -> flask
_wcam_result   = {"objects": {}, "details": [], "seq": 0}
_result_lock   = threading.Lock()

def _yolo_process(in_q, out_q):
    from ultralytics import YOLO
    import numpy as np
    m = YOLO("yolov8m-oiv7.pt")
    m.predict(source=np.zeros((640,640,3),dtype=np.uint8), imgsz=640, conf=0.08, verbose=False)
    print("[NexVision] Webcam YOLO process ready!")
    while True:
        item = in_q.get()
        if item is None: break
        frame, conf = item
        try:
            use_conf = max(0.08, min(conf, 0.30))
            res = m.predict(
                source=frame, conf=use_conf, iou=0.30,
                max_det=500, imgsz=640,
                agnostic_nms=True, verbose=False
            )[0]
            objects, details = {}, []
            for box in res.boxes:
                cid=int(box.cls[0]); label=m.names[cid]; score=float(box.conf[0])
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                objects[label]=objects.get(label,0)+1
                details.append({"label":label,"conf":round(score*100,1),"box":[x1,y1,x2,y2]})
            while not out_q.empty():
                try: out_q.get_nowait()
                except: break
            out_q.put({"objects":objects,"details":details})
        except Exception as e:
            print(f"[YOLO] Error: {e}")

def _result_collector():
    """Thread that reads results from the process queue into _wcam_result."""
    while True:
        try:
            r = _wcam_out.get(timeout=1)
            with _result_lock:
                _wcam_result["objects"] = r["objects"]
                _wcam_result["details"] = r["details"]
                _wcam_result["seq"]    += 1
        except: pass

_latest_frame = None
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()

def _frame_feeder():
    """Always sends latest frame to YOLO process, replaces old one if busy."""
    while True:
        _frame_event.wait()
        _frame_event.clear()
        with _frame_lock:
            item = _latest_frame
        if item is None: continue
        # drain queue first so we always send the freshest frame
        while not _wcam_in.empty():
            try: _wcam_in.get_nowait()
            except: break
        try: _wcam_in.put_nowait(item)
        except: pass

def load_model_bg():
    global model, webcam_model, model_ready, _wcam_in, _wcam_out
    from ultralytics import YOLO
    _wcam_in  = mp.Queue(maxsize=1)
    _wcam_out = mp.Queue(maxsize=2)
    p = mp.Process(target=_yolo_process, args=(_wcam_in, _wcam_out), daemon=True)
    p.start()
    threading.Thread(target=_result_collector, daemon=True).start()
    threading.Thread(target=_frame_feeder,     daemon=True).start()
    model = YOLO("yolov8m-oiv7.pt")
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model.predict(source=dummy, conf=0.08, verbose=False)
    model_ready = True
    print("[NexVision] AI Model ready!")

threading.Thread(target=load_model_bg, daemon=True).start()

np.random.seed(42)
COLORS = {i: tuple(int(x) for x in np.random.randint(60, 255, 3)) for i in range(601)}
# pre-build color map so all 601 classes have unique distinct colors
for i in range(601):
    h = (i * 47) % 360
    s, v = 0.85, 0.95
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h/360, s, v)
    COLORS[i] = (int(b*255), int(g*255), int(r*255))

SCENES = {
    "Traffic":     {"car","truck","bus","motorcycle","bicycle","traffic light","stop sign"},
    "Kitchen":     {"bottle","cup","fork","knife","spoon","bowl","banana","apple","pizza","donut","cake","microwave","oven","refrigerator"},
    "Office":      {"laptop","keyboard","mouse","cell phone","book","clock","chair","tv"},
    "Sports":      {"sports ball","frisbee","skateboard","surfboard","tennis racket","baseball bat"},
    "Animals":     {"cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","bird"},
    "Living Room": {"couch","tv","remote","potted plant","vase","chair"},
}
DANGER   = {"knife","scissors"}
VEHICLES = {"car","truck","bus","motorcycle","bicycle","boat","airplane","train"}
ANIMALS  = {"cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","bird"}

# ── AI Chatbot ────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def chatbot_reply(msg, last_objects=None, page=None):
    m = msg.lower().strip()
    objs = last_objects or {}

    # ── GREETINGS ──
    if any(w in m for w in ["hi","hello","hey","hola","sup","yo"]):
        return "Hey! 👋 I'm your NexVision Assistant. Ask me anything about this website — how to detect objects, use the webcam, view history, understand your dashboard stats, and more!"

    # ── WHAT IS THIS WEBSITE ──
    if any(w in m for w in ["what is this","what is nexvision","about this","about website","what does this","purpose","explain website","what can this website"]):
        return "NexVision is an AI-powered object detection platform. You can upload images or videos, use your live webcam, and the AI will detect and label every object it sees in real-time using YOLOv8 — one of the most powerful detection models available. It supports 600+ object classes!"

    # ── HOW TO USE / GET STARTED ──
    if any(w in m for w in ["how to use","how do i use","get started","how to start","where to start","guide","tutorial","steps"]):
        return "Here's how to get started:<br><br>① Go to <b>Detect</b> in the top menu<br>② Choose <b>Image</b>, <b>Video</b>, or <b>Webcam</b> mode<br>③ Upload your file or start the camera<br>④ Adjust the <b>Confidence</b> slider (lower = detect more)<br>⑤ Click <b>Run Detection</b> — results appear instantly!<br><br>All your detections are saved in <b>History</b> automatically."

    # ── DETECT PAGE ──
    if any(w in m for w in ["detect page","detection page","how to detect","run detection","start detection","detect objects"]):
        return "On the <b>Detect</b> page you can:<br>• Upload an <b>image</b> (PNG, JPG, WEBP, BMP)<br>• Upload a <b>video</b> (MP4, AVI, MOV, MKV)<br>• Use your <b>live webcam</b> for real-time detection<br><br>After detection you'll see bounding boxes, object labels, confidence scores, and an AI scene analysis!"

    # ── WEBCAM ──
    if any(w in m for w in ["webcam","camera","live","real time","realtime","live detection"]):
        return "To use <b>Webcam mode</b>:<br>① Go to <b>Detect</b> page<br>② Click the <b>Webcam</b> tab<br>③ Click <b>Start Camera</b> and allow browser permission<br>④ The AI detects objects in real-time automatically!<br><br>💡 Tip: Use <b>Chrome or Edge</b> at <b>http://127.0.0.1:8000</b> for best results. Good lighting improves accuracy."

    # ── CONFIDENCE SLIDER ──
    if any(w in m for w in ["confidence","slider","threshold","sensitivity","accuracy","detect more","detect less"]):
        return "The <b>Confidence Slider</b> controls how certain the AI must be before labeling an object:<br><br>• <b>Low (10-20%)</b> — detects more objects, may include uncertain ones<br>• <b>Medium (25-40%)</b> — balanced, recommended for most use<br>• <b>High (50%+)</b> — only very confident detections<br><br>If objects aren't being detected, try sliding it <b>lower</b>!"

    # ── SUPPORTED FILES ──
    if any(w in m for w in ["supported","file type","format","what files","upload","image format","video format"]):
        return "NexVision supports:<br><br>🖼️ <b>Images:</b> PNG, JPG, JPEG, BMP, WEBP<br>🎥 <b>Videos:</b> MP4, AVI, MOV, MKV<br>📹 <b>Live Webcam:</b> Any browser camera<br><br>Max file size: <b>200MB</b>"

    # ── DASHBOARD ──
    if any(w in m for w in ["dashboard","stats","statistics","total scans","total objects","avg confidence","charts","graph"]):
        return "Your <b>Dashboard</b> shows:<br><br>🎯 <b>Total Scans</b> — how many detections you've run<br>🔍 <b>Objects Detected</b> — total objects found across all scans<br>🏷️ <b>Unique Classes</b> — how many different object types detected<br>⚡ <b>Avg Confidence</b> — average detection confidence<br>📊 <b>Charts</b> — top objects and daily activity graphs<br>🕒 <b>Recent Detections</b> — your last 6 scans"

    # ── HISTORY ──
    if any(w in m for w in ["history","past detection","previous","saved","my detection","delete"]):
        return "The <b>History</b> page shows all your past detections with:<br><br>• Original filename and media type<br>• All detected objects and counts<br>• Detection date and time<br>• Confidence and processing time<br><br>You can <b>delete</b> any detection by clicking the delete button on it."

    # ── WHAT CAN IT DETECT ──
    if any(w in m for w in ["what can","detect gun","detect phone","detect mouse","classes","600","objects list","what objects","can it detect"]):
        return "NexVision can detect <b>600+ object classes</b> including:<br><br>🔫 Handgun, Shotgun, Rifle, Sword, Knife<br>📱 Mobile phone, Laptop, Computer mouse, Keyboard<br>👤 Person, Face, Clothing<br>🚗 Car, Truck, Motorcycle, Bicycle<br>🐶 Dog, Cat, Bird and many animals<br>🍎 Food, Furniture, Sports equipment<br><br>If something isn't detected, lower the confidence slider to <b>15%</b>!"

    # ── REGISTER / LOGIN ──
    if any(w in m for w in ["register","sign up","create account","login","log in","sign in","password","account"]):
        return "To use NexVision:<br><br>• <b>Register</b> — click <b>Get Started</b> on the home page, enter username, email and password<br>• <b>Login</b> — use your email and password<br>• <b>Logout</b> — click Logout in the top navigation<br><br>Your account saves all your detection history and stats!"

    # ── NAVIGATION ──
    if any(w in m for w in ["navigation","menu","pages","where is","how to go","go to"]):
        return "NexVision has these pages:<br><br>🏠 <b>Home</b> — landing page<br>📊 <b>Dashboard</b> — your stats and recent detections<br>🔍 <b>Detect</b> — run object detection<br>📜 <b>History</b> — all past detections<br><br>All accessible from the <b>top navigation bar</b>."

    # ── AI / MODEL ──
    if any(w in m for w in ["yolo","model","ai","how does it work","algorithm","neural","machine learning","deep learning"]):
        return "NexVision uses <b>YOLOv8</b> (You Only Look Once) — a state-of-the-art deep learning model.<br><br>• Trained on <b>Open Images V7</b> dataset with 600+ classes<br>• Detects multiple objects <b>simultaneously</b> in one pass<br>• Draws <b>bounding boxes</b> with confidence scores<br>• Webcam uses a separate process so video stays smooth"

    # ── SCENE ANALYSIS ──
    if any(w in m for w in ["scene","analysis","ai analysis","insights","alert"]):
        return "After each detection, NexVision shows an <b>AI Scene Analysis</b> with:<br><br>🎭 <b>Scene type</b> — Office, Kitchen, Traffic, etc.<br>👤 <b>Crowd level</b> — how many people detected<br>⚠️ <b>Alerts</b> — dangerous objects, large crowds, weapons<br>🏆 <b>Dominant object</b> — most detected item<br>📊 <b>Total counts</b> — objects and classes"

    # ── LAST DETECTION OBJECTS ──
    if any(w in m for w in ["what did you detect","what do you see","last detection","what was detected","show objects"]):
        if objs:
            items = ", ".join(f"<b>{v}x {k}</b>" for k,v in sorted(objs.items(), key=lambda x:-x[1]))
            return f"Last detection found: {items} — Total {sum(objs.values())} objects across {len(objs)} classes!"
        return "No detection has been run yet in this session. Go to <b>Detect</b> and run a detection first!"

    # ── TIPS ──
    if any(w in m for w in ["tip","help","improve","better result","suggestion","advice","not detecting","not working"]):
        return "💡 <b>Tips for best results:</b><br><br>• Use <b>good lighting</b> — dark images reduce accuracy<br>• <b>Lower confidence</b> slider to 15% if objects aren't detected<br>• Hold objects <b>clearly facing</b> the camera<br>• Use <b>Chrome or Edge</b> browser for webcam<br>• For small objects, get <b>closer</b> to the camera<br>• Make sure the <b>AI model is ready</b> (yellow banner disappears)"

    # ── FEEDBACK ──
    if any(w in m for w in ["feedback","rating","rate","review"]):
        return "You can submit <b>feedback</b> on any detection result! After running a detection, a feedback option lets you rate the accuracy and leave a comment. This helps improve the system!"

    # ── THANK YOU ──
    if any(w in m for w in ["thank","thanks","great","awesome","good job","nice","cool","love it"]):
        return "You're welcome! 😊 NexVision is here to make object detection easy and powerful. Let me know if you need anything else!"

    # ── BYE ──
    if any(w in m for w in ["bye","goodbye","exit","see you","later"]):
        return "Goodbye! 👋 Come back anytime. Happy detecting!"

    # ── DEFAULT ──
    return f"I'm not sure about that, but I can help with:<br><br>• How to use the website<br>• How detection works<br>• Dashboard and history<br>• Webcam and file upload<br>• What objects can be detected<br>• Tips for better accuracy<br><br>Just ask me anything! 😊"

# ── Scene analysis ────────────────────────────────────────────────────────────
def ai_analyze(objects):
    labels = set(objects.keys())
    scene, best = "General", 0
    for s, kw in SCENES.items():
        sc = len(labels & kw)
        if sc > best: scene, best = s, sc
    people  = objects.get("person", 0)
    crowd   = "None" if people==0 else "Low" if people<=2 else "Medium" if people<=6 else f"High ({people})"
    alerts  = []
    if labels & DANGER:   alerts.append(f"Dangerous object: {', '.join(labels & DANGER)}")
    if people > 8:        alerts.append(f"Large crowd: {people} people")
    if labels & ANIMALS:  alerts.append(f"Animals: {', '.join(labels & ANIMALS)}")
    if labels & VEHICLES: alerts.append(f"Vehicles: {', '.join(labels & VEHICLES)}")
    dominant = max(objects, key=objects.get) if objects else "none"
    return {
        "scene": scene, "crowd": crowd, "alerts": alerts, "dominant": dominant,
        "total": sum(objects.values()), "classes": len(objects),
        "insights": [
            f"Scene: <b>{scene}</b>",
            f"Dominant: <b>{dominant}</b> x{objects.get(dominant,0)}",
            f"Total: <b>{sum(objects.values())}</b> objects in <b>{len(objects)}</b> classes",
            f"People: <b>{crowd}</b>",
        ]
    }

# ── Detection ─────────────────────────────────────────────────────────────────
def run_detection(bgr, conf=0.10):
    results = model.predict(
        source=bgr, conf=conf, iou=0.30, max_det=500,
        agnostic_nms=True, verbose=False
    )[0]
    objects = {}
    details = []
    out = bgr.copy()
    H, W = out.shape[:2]
    for box in results.boxes:
        cid   = int(box.cls[0])
        label = model.names[cid]
        score = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(W,x2), min(H,y2)
        col = COLORS[cid % 601]
        cv2.rectangle(out,(x1,y1),(x2,y2),col,2)
        L=12
        for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(px,py),(px+dx*L,py),col,3)
            cv2.line(out,(px,py),(px,py+dy*L),col,3)
        txt = f"{label} {score:.0%}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1-th-6, 0)
        cv2.rectangle(out,(x1,ly),(x1+tw+8,ly+th+8),col,-1)
        cv2.putText(out,txt,(x1+4,ly+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
        objects[label] = objects.get(label,0)+1
        details.append({"label":label,"conf":round(score*100,1),"box":[x1,y1,x2,y2]})
    return out, objects, details

def img_to_b64(bgr, quality=85):
    buf = io.BytesIO()
    Image.fromarray(bgr[:,:,::-1]).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_bgr(data):
    if "," in data: data = data.split(",")[1]
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def read_img(path):
    pil = Image.open(path).convert("RGB")
    return np.array(pil)[:,:,::-1].copy()

def save_det(uid, orig, resf, objects, mtype, conf, ptime):
    db  = get_db()
    res = db.detections.insert_one({
        "user_id": uid, "original_filename": orig, "result_filename": resf,
        "objects_json": objects, "total_objects": sum(objects.values()),
        "unique_objects": len(objects), "media_type": mtype,
        "confidence": conf, "processing_time": ptime,
        "created_at": datetime.now(timezone.utc)
    })
    for obj, cnt in objects.items():
        db.object_stats.update_one(
            {"user_id": uid, "object_name": obj},
            {"$inc": {"detect_count": cnt}, "$set": {"last_seen": datetime.now(timezone.utc)}},
            upsert=True
        )
    return str(res.inserted_id)

# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        u=request.form["username"].strip(); e=request.form["email"].strip()
        p=generate_password_hash(request.form["password"])
        try:
            get_db().users.insert_one({
                "username": u, "email": e, "password": p,
                "avatar": u[0].upper(), "created_at": datetime.now(timezone.utc)
            })
            flash("Account created! Login now.","success")
            return redirect(url_for("login"))
        except: flash("Username or email already exists.","error")
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        e=request.form["email"].strip(); p=request.form["password"]
        u=get_db().users.find_one({"email": e})
        if u and check_password_hash(u["password"], p):
            session.update({"user_id": str(u["_id"]), "username": u["username"], "avatar": u["avatar"]})
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.","error")
    return render_template("login.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("index"))

# ── CHATBOT ───────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    msg  = data.get("message","").strip()
    last_objects = data.get("last_objects", {})
    page = data.get("page", "")
    if not msg: return jsonify({"reply":"Please type a message!"})
    reply = chatbot_reply(msg, last_objects, page)
    return jsonify({"reply": reply})

# ── MODEL STATUS ──────────────────────────────────────────────────────────────
@app.route("/api/model_status")
def api_model_status():
    return jsonify({"ready": model_ready})

@app.route("/api/wcam_result")
def api_wcam_result():
    if "user_id" not in session: return jsonify({}), 401
    with _result_lock:
        r = dict(_wcam_result)
    return jsonify({"objects": r["objects"], "details": r["details"],
                    "total": sum(r["objects"].values()), "seq": r["seq"]})

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session: return redirect(url_for("login"))
    uid = session["user_id"]
    db  = get_db()
    recent = list(db.detections.find({"user_id": uid}).sort("created_at", -1).limit(6))
    for r in recent:
        r["id"] = str(r["_id"])
        if isinstance(r.get("objects_json"), str):
            try: r["objects_json"] = json.loads(r["objects_json"])
            except: r["objects_json"] = {}
    agg   = list(db.detections.aggregate([
        {"$match": {"user_id": uid}},
        {"$group": {"_id": None, "ts": {"$sum": 1}, "to": {"$sum": "$total_objects"},
                    "uo": {"$sum": "$unique_objects"}, "ac": {"$avg": "$confidence"}}}
    ]))
    ts = agg[0]["ts"] if agg else 0
    to = agg[0]["to"] if agg else 0
    uo = agg[0]["uo"] if agg else 0
    ac = round((agg[0]["ac"] or 0) * 100, 1) if agg else 0
    top = list(db.object_stats.find({"user_id": uid}).sort("detect_count", -1).limit(5))
    return render_template("dashboard.html", recent=recent, total_scans=ts, total_objects=to,
                           unique_objects=uo, top_objects=top, avg_conf=ac)

# ── DETECT ────────────────────────────────────────────────────────────────────
@app.route("/detect", methods=["GET","POST"])
def detect():
    if "user_id" not in session: return redirect(url_for("login"))
    if request.method=="POST":
        if not model_ready:
            return jsonify({"error":"Model loading, please wait..."}),503
        conf = float(request.form.get("confidence",0.30))
        mode = request.form.get("mode","image")

        if mode=="webcam":
            raw=request.form.get("frame_data","")
            if not raw: return jsonify({"error":"no frame"}),400
            try: frame=b64_to_bgr(raw)
            except Exception as ex: return jsonify({"error":str(ex)}),400
            conf2 = float(request.form.get("confidence", 0.30))
            with _frame_lock:
                global _latest_frame
                _latest_frame = (frame, conf2)
            _frame_event.set()
            return jsonify({"ok": True})

        if "file" not in request.files: return jsonify({"error":"no file"}),400
        f=request.files["file"]
        if not f or not ("." in f.filename and f.filename.rsplit(".",1)[-1].lower() in ALLOWED_IMG|ALLOWED_VID):
            return jsonify({"error":"invalid file"}),400

        fname=f"{int(time.time())}_{secure_filename(f.filename)}"
        fpath=os.path.join(UPLOAD_FOLDER,fname)
        f.save(fpath); t0=time.time()

        if f.filename.rsplit(".",1)[-1].lower() in ALLOWED_VID:
            rname="result_"+fname; rpath=os.path.join(RESULT_FOLDER,rname)
            cap=cv2.VideoCapture(fpath)
            fps=cap.get(cv2.CAP_PROP_FPS) or 25
            W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # resize large videos for faster processing
            scale = 1.0
            if W > 1280: scale = 1280/W
            rW, rH = int(W*scale), int(H*scale)
            writer=cv2.VideoWriter(rpath,cv2.VideoWriter_fourcc(*"mp4v"),fps,(rW,rH))
            all_obj={}; fc=0; first=None; skip=2  # process every 2nd frame
            last_out=None; last_objs={}
            while cap.isOpened():
                ret,frame=cap.read()
                if not ret: break
                if scale < 1.0: frame=cv2.resize(frame,(rW,rH))
                if fc % skip == 0:  # run YOLO on every 2nd frame
                    last_out,last_objs,_=run_detection(frame,conf)
                else:
                    last_out = last_out if last_out is not None else frame
                if fc==0: first=last_out.copy()
                for k,v in last_objs.items(): all_obj[k]=all_obj.get(k,0)+v
                writer.write(last_out); fc+=1
            cap.release(); writer.release()
            ptime=round(time.time()-t0,3)
            thumb=img_to_b64(first) if first is not None else None
            did=save_det(session["user_id"],fname,rname,all_obj,"video",conf,ptime)
            return jsonify({"objects":all_obj,"total":sum(all_obj.values()),"result_image":thumb,
                            "proc_time":ptime,"frames":fc,"detection_id":did,"media_type":"video","ai":ai_analyze(all_obj)})

        try: frame=read_img(fpath)
        except: return jsonify({"error":"cannot read image"}),400
        out,objects,details=run_detection(frame,conf)
        rname="result_"+fname
        Image.fromarray(out[:,:,::-1]).save(os.path.join(RESULT_FOLDER,rname),quality=90)
        ptime=round(time.time()-t0,3)
        did=save_det(session["user_id"],fname,rname,objects,"image",conf,ptime)
        return jsonify({"objects":objects,"details":details,"total":sum(objects.values()),
                        "result_image":img_to_b64(out,90),"proc_time":ptime,"detection_id":did,"media_type":"image","ai":ai_analyze(objects)})

    return render_template("detect.html")

# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.route("/history")
def history():
    if "user_id" not in session: return redirect(url_for("login"))
    records = list(get_db().detections.find({"user_id": session["user_id"]}).sort("created_at", -1))
    for r in records:
        r["id"] = str(r["_id"])
        if isinstance(r.get("objects_json"), str):
            try: r["objects_json"] = json.loads(r["objects_json"])
            except: r["objects_json"] = {}
    return render_template("history.html", records=records)

@app.route("/delete_detection/<string:did>", methods=["POST"])
def delete_detection(did):
    if "user_id" not in session: return jsonify({"error":"unauth"}),401
    get_db().detections.delete_one({"_id": ObjectId(did), "user_id": session["user_id"]})
    return jsonify({"success": True})

@app.route("/api/stats")
def api_stats():
    if "user_id" not in session: return jsonify({}),401
    uid = session["user_id"]
    db  = get_db()
    top = list(db.object_stats.find({"user_id": uid}).sort("detect_count", -1).limit(10))
    daily = list(db.detections.aggregate([
        {"$match": {"user_id": uid}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": -1}}, {"$limit": 7}
    ]))
    return jsonify({
        "top_objects": [{"object_name": t["object_name"], "detect_count": t["detect_count"]} for t in top],
        "daily": [{"date": d["_id"], "count": d["count"]} for d in daily]
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    if "user_id" not in session: return jsonify({"error":"unauth"}),401
    d = request.json
    get_db().feedback.insert_one({
        "user_id": session["user_id"], "detection_id": d.get("detection_id"),
        "rating": d.get("rating", 5), "comment": d.get("comment", ""),
        "created_at": datetime.now(timezone.utc)
    })
    return jsonify({"success": True})

@app.route("/results/<filename>")
def serve_result(filename): return send_from_directory(RESULT_FOLDER,filename)

if __name__=="__main__":
    mp.freeze_support()
    init_db()
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:8000')).start()
    print("[NexVision] Opening http://127.0.0.1:8000")
    from waitress import serve
    print("[NexVision] Running on waitress (multi-threaded production server)")
    serve(app, host='127.0.0.1', port=8000, threads=8)