#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           AAYAN GLOBE  v1.0  —  Python / PyQt5 + OpenGL     ║
║                                                              ║
║  pip install PyQt5 PyOpenGL PyOpenGL_accelerate numpy        ║
║  pip install requests                                         ║
║  python3 aayan_globe.py                                      ║
╚══════════════════════════════════════════════════════════════╝

Architecture (fixes 0% stuck bug):
  • GlobeWidget is always visible (GL context initialises immediately)
  • LoadOverlay sits on top as a plain QWidget (not in QStackedWidget)
  • Heavy numpy work runs in BuildThread so UI stays responsive
  • initializeGL() uploads data to GPU as soon as both:
      (a) GL context is ready  AND  (b) arrays are built
"""

import sys, math, time, ctypes
from datetime import datetime
import numpy as np

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QOpenGLWidget,
        QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
    from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QPalette, QSurfaceFormat
except ImportError:
    sys.exit("PyQt5 not found.  pip install PyQt5")

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    sys.exit("PyOpenGL not found.  pip install PyOpenGL PyOpenGL_accelerate")

try:
    import requests as _req; HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ═══════════════════════════════════════════════════════════════
# COLOURS
# ═══════════════════════════════════════════════════════════════
C_BG      = QColor(1,4,9)
C_PANEL   = QColor(5,12,28,235)
C_BORDER  = QColor(13,33,55)
C_ACCENT  = QColor(0,255,231)
C_ACCENT2 = QColor(255,45,107)
C_ACCENT3 = QColor(255,195,0)
C_DIM     = QColor(42,64,96)
C_TEXT    = QColor(160,196,224)

def _c(q): return f"rgb({q.red()},{q.green()},{q.blue()})"
def _ca(q,a): return f"rgba({q.red()},{q.green()},{q.blue()},{a})"

PANEL_CSS  = f"background:{_ca(C_PANEL,235)};border:1px solid {_c(C_BORDER)};"
BTN_BASE   = (f"QPushButton{{background:transparent;border:1px solid {_c(C_DIM)};"
              f"color:{_c(C_DIM)};font-family:'Courier New';font-size:8pt;"
              f"letter-spacing:2px;padding:3px 8px;}}"
              f"QPushButton:hover{{border-color:{_c(C_TEXT)};color:{_c(C_TEXT)};}}")

def togbtn(active, color=None):
    color = color or C_ACCENT
    if active:
        return (f"QPushButton{{background:rgba({color.red()},{color.green()},{color.blue()},22);"
                f"border:1px solid {_c(color)};color:{_c(color)};"
                f"font-family:'Courier New';font-size:8pt;letter-spacing:2px;padding:3px 8px;}}")
    return BTN_BASE

# ═══════════════════════════════════════════════════════════════
# MATHS
# ═══════════════════════════════════════════════════════════════
R = 1.0

def ll2xyz(lat, lng, r=R):
    phi   = math.radians(90 - lat)
    theta = math.radians(lng + 180)
    return (-r*math.sin(phi)*math.cos(theta),
             r*math.cos(phi),
             r*math.sin(phi)*math.sin(theta))

def _norm(v):
    l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return (v[0]/l,v[1]/l,v[2]/l) if l else v

# ═══════════════════════════════════════════════════════════════
# BUILD THREAD  — all numpy work, never touches GL
# ═══════════════════════════════════════════════════════════════
class BuildThread(QThread):
    progress = pyqtSignal(int, str)          # pct, message
    done     = pyqtSignal(dict)              # arrays dict

    CONTINENTS = [
        (-168,-52,7,83,[(-168,70),(-140,60),(-125,49),(-95,28),(-85,10),(-77,7),
          (-52,10),(-52,23),(-70,43),(-60,47),(-52,56),(-60,63),(-80,73),(-100,83),(-140,72),(-168,70)]),
        (-55,-17,60,84,[(-55,60),(-40,60),(-20,65),(-17,75),(-20,84),(-35,84),(-50,78),(-55,68),(-55,60)]),
        (-82,-34,-56,13,[(-82,13),(-60,13),(-50,5),(-35,-10),(-34,-8),(-40,-20),
          (-48,-28),(-52,-34),(-63,-56),(-70,-50),(-75,-40),(-80,-30),(-82,-5),(-82,13)]),
        (-10,25,36,60,[(-10,36),(10,36),(15,38),(25,37),(25,42),(20,44),(18,48),
          (10,52),(5,54),(0,54),(-2,56),(-5,48),(-10,44),(-10,36)]),
        (5,30,57,71,[(5,57),(15,57),(20,60),(25,65),(22,68),(26,70),(20,71),(15,70),(10,65),(5,62),(5,57)]),
        (25,45,36,60,[(25,37),(40,37),(45,42),(45,48),(40,52),(35,55),(30,55),(25,52),(25,45),(25,37)]),
        (-18,52,-35,38,[(-18,15),(-5,5),(0,-5),(10,-5),(15,-20),(18,-30),(20,-35),
          (28,-35),(33,-30),(40,-20),(45,-10),(52,10),(44,12),(42,15),(38,22),(32,31),
          (25,37),(15,38),(5,37),(-5,35),(-5,28),(-18,20),(-18,15)]),
        (44,51,-26,-12,[(44,-13),(46,-12),(50,-15),(51,-18),(50,-22),(47,-25),(44,-26),(44,-13)]),
        (25,90,1,75,[(25,37),(40,37),(50,30),(58,22),(58,12),(50,8),(45,1),(50,2),
          (60,10),(70,23),(80,28),(88,28),(90,23),(90,72),(70,72),(55,68),(40,55),(30,50),(25,45),(25,37)]),
        (90,145,1,75,[(90,23),(95,22),(100,5),(104,1),(110,1),(115,3),(118,22),
          (122,30),(122,37),(130,45),(135,50),(145,50),(145,75),(130,73),(100,73),(90,72),(90,23)]),
        (130,145,31,45,[(130,31),(131,33),(132,34),(133,35),(135,36),(137,37),(140,40),
          (141,41),(142,43),(143,44),(145,44),(143,42),(141,40),(139,36),(137,34),(134,33),(131,32),(130,31)]),
        (65,92,8,37,[(65,23),(68,23),(70,20),(72,22),(74,8),(78,8),(80,10),(82,8),(80,13),
          (82,20),(85,22),(88,22),(90,24),(92,26),(92,27),(88,28),(84,24),(80,28),(72,28),(65,23)]),
        (97,110,1,28,[(97,18),(100,20),(100,25),(104,22),(104,18),(102,12),(100,5),(104,1),
          (108,1),(110,5),(108,12),(105,18),(100,26),(97,24),(97,18)]),
        (95,119,-9,6,[(95,5),(100,5),(104,1),(108,-5),(112,-8),(116,-9),(119,-8),(119,-5),
          (115,-3),(110,1),(106,-8),(102,-5),(98,2),(95,5)]),
        (118,127,5,19,[(118,10),(120,10),(122,5),(124,7),(126,8),(127,10),(126,12),(125,17),(122,19),(120,16),(118,10)]),
        (114,154,-44,-10,[(114,-22),(122,-18),(130,-12),(136,-12),(140,-18),(148,-20),
          (154,-28),(152,-38),(146,-44),(138,-44),(130,-33),(120,-34),(114,-30),(114,-22)]),
        (166,178,-47,-34,[(166,-46),(168,-46),(170,-44),(172,-43),(172,-40),(174,-37),
          (174,-34),(176,-36),(178,-38),(178,-42),(174,-44),(170,-46),(166,-46)]),
        (-8,2,50,59,[(-8,52),(-5,52),(-3,50),(0,50),(2,52),(1,55),(-2,58),(-5,58),(-6,56),(-5,54),(-8,53),(-8,52)]),
        (-25,-12,63,67,[(-25,63),(-20,63),(-14,65),(-12,65),(-15,66),(-20,67),(-25,66),(-25,63)]),
        (-180,180,-90,-65,[(-180,-90),(180,-90),(180,-65),(-180,-65),(-180,-90)]),
        (45,90,55,75,[(45,55),(60,55),(70,55),(80,57),(90,60),(90,72),(70,72),(55,68),(45,65),(45,55)]),
        (130,145,42,60,[(130,42),(135,45),(140,50),(142,55),(145,58),(142,60),(138,58),(135,54),(132,48),(130,44),(130,42)]),
    ]

    def run(self):
        # 1. Land mask
        self.progress.emit(5, "Building land mask…")
        mask = self._build_mask()
        self.progress.emit(30, "Land mask done")

        # 2. Dot arrays
        self.progress.emit(32, "Generating dot field (low LOD)…")
        dots_low  = self._dots(mask, 3.0)
        self.progress.emit(45, "Generating dot field (medium LOD)…")
        dots_med  = self._dots(mask, 2.0)
        self.progress.emit(58, "Generating dot field (high LOD)…")
        dots_high = self._dots(mask, 1.2)
        self.progress.emit(70, "Dot fields ready")

        # 3. Borders
        self.progress.emit(72, "Building country borders…")
        borders = self._borders(mask)
        self.progress.emit(87, "Borders ready")

        # 4. Stars
        self.progress.emit(89, "Building star field…")
        stars = self._stars()
        self.progress.emit(95, "All arrays ready — uploading to GPU…")

        self.done.emit({
            "dots":   {"low": dots_low, "med": dots_med, "high": dots_high},
            "border": borders,
            "stars":  stars,
        })

    def _build_mask(self):
        mask = np.zeros((360,180), dtype=bool)
        def pip(x, y, poly):
            inside = False; ox,oy = poly[-1]
            for cx,cy in poly:
                if ((cy>y)!=(oy>y)) and (x < (ox-cx)*(y-cy)/(oy-cy+1e-12)+cx):
                    inside = not inside
                ox,oy = cx,cy
            return inside
        for ml,xl,mb,xb,poly in self.CONTINENTS:
            for lng in range(max(-180,int(ml)), min(180,int(xl)+1)):
                for lat in range(max(-90,int(mb)), min(90,int(xb)+1)):
                    li=(lng+180)%360; la=max(0,min(179,89-lat))
                    if mask[li,la]: continue
                    if pip(lng+0.5, lat+0.5, poly): mask[li,la]=True
        return mask

    def _dots(self, mask, res):
        rows = []
        for lat in np.arange(-90, 91, res):
            for lng in np.arange(-180, 180, res):
                li = int((lng+180)%360) % 360
                la = max(0, min(179, int(89-lat)))
                x,y,z = ll2xyz(lat,lng)
                if mask[li,la]:
                    rows.append((x,y,z, 0.0,1.0,0.906))
                else:
                    rows.append((x,y,z, 0.0,0.2,0.733))
        return np.array(rows, dtype=np.float32)

    def _borders(self, mask):
        segs = []; res = 1.5
        for lat in np.arange(-88, 89, res):
            for lng in np.arange(-180, 179, res):
                li  = int((lng+180)%360)%360
                la  = max(0,min(179,int(89-lat)))
                li2 = (li+1)%360
                la2 = max(0,min(179,la+1))
                cur = mask[li,la]
                if cur != mask[li2,la]:
                    segs += [ll2xyz(lat,lng,R+0.005), ll2xyz(lat+res,lng,R+0.005)]
                if cur != mask[li,la2]:
                    segs += [ll2xyz(lat,lng,R+0.005), ll2xyz(lat,lng+res,R+0.005)]
        return np.array(segs, dtype=np.float32) if segs else np.zeros((2,3),dtype=np.float32)

    def _stars(self):
        rng = np.random.default_rng(42); rows = []
        for _ in range(4000):
            theta = rng.uniform(0,2*math.pi)
            phi   = math.acos(2*rng.uniform()-1)
            r     = rng.uniform(9,12); b = rng.uniform(0.3,1.0)
            x = r*math.sin(phi)*math.cos(theta)
            y = r*math.cos(phi)
            z = r*math.sin(phi)*math.sin(theta)
            rows.append((x,y,z, b*0.85,b*0.9,b))
        return np.array(rows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# GEOLOCATION THREAD
# ═══════════════════════════════════════════════════════════════
class GeoThread(QThread):
    result  = pyqtSignal(dict)
    message = pyqtSignal(str,str,str)
    APIS = [
        ("https://ipwho.is/",
         lambda d: dict(ip=d.get("ip",""),isp=d.get("connection",{}).get("isp","") or d.get("org",""),
                        city=d.get("city",""),region=d.get("region",""),country=d.get("country",""),
                        lat=float(d.get("latitude",0)),lng=float(d.get("longitude",0)),
                        ok=d.get("success",True) is not False)),
        ("https://ip-api.com/json/?fields=status,message,country,regionName,city,lat,lon,isp,org,query",
         lambda d: dict(ip=d.get("query",""),isp=d.get("org","") or d.get("isp",""),
                        city=d.get("city",""),region=d.get("regionName",""),country=d.get("country",""),
                        lat=float(d.get("lat",0)),lng=float(d.get("lon",0)),
                        ok=d.get("status","")=="success")),
        ("https://ipapi.co/json/",
         lambda d: dict(ip=d.get("ip",""),isp=d.get("org","") or d.get("isp",""),
                        city=d.get("city",""),region=d.get("region",""),
                        country=d.get("country_name","") or d.get("country",""),
                        lat=float(d.get("latitude",0)),lng=float(d.get("longitude",0)),
                        ok=not d.get("error"))),
    ]
    def run(self):
        if not HAS_REQUESTS:
            self.message.emit("warn","REQUESTS","requests not installed — using default")
            self.result.emit(dict(ip="--",isp="--",city="Jharkhand",region="Jharkhand",
                                  country="India",lat=23.35,lng=85.33,source="default")); return
        for url,parse in self.APIS:
            host=url.split("/")[2]; self.message.emit("info","IP: "+host,"Trying…")
            try:
                r=_req.get(url,timeout=6,headers={"User-Agent":"AayanGlobe/1.0"}); r.raise_for_status()
                d=parse(r.json())
                if d["ok"] and d["lat"] and d["lng"]: d["source"]="ip"; self.result.emit(d); return
            except Exception as e: self.message.emit("warn","FAIL",str(e))
        self.result.emit(dict(ip="--",isp="--",city="Jharkhand",region="Jharkhand",
                              country="India",lat=23.35,lng=85.33,source="default"))


# ═══════════════════════════════════════════════════════════════
# GLOBE OPENGL WIDGET
# ═══════════════════════════════════════════════════════════════
class GlobeWidget(QOpenGLWidget):
    """
    Always visible.  Arrays arrive via set_arrays() from BuildThread.
    If GL is already ready when arrays arrive, uploads immediately.
    If GL isn't ready yet, initializeGL() will upload when it runs.
    """
    glReady = pyqtSignal()

    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setSamples(4); fmt.setDepthBufferSize(24)
        fmt.setVersion(2,1); fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        super().__init__(parent)
        self.setFormat(fmt)

        # Camera
        self.cam_z    = 2.9
        self.rot_x    = 0.0    # degrees
        self.rot_y    = 0.0    # radians
        self.vel_x    = self.vel_y = 0.0
        self.dragging = False
        self.last_pos = QPoint()

        # Settings
        self.rotation_dir = "west-east"
        self.sun_enabled  = True
        self.moon_enabled = False
        self.bg_affection = True

        # Marker
        self.marker_lat = self.marker_lng = None
        self.marker_source = "ip"

        # Moon
        self.moon_angle = 0.0

        # Star rotation (follows globe)
        self.star_rot_x = self.star_rot_y = 0.0

        # GL handles
        self._vbos   = {}     # "dots_low/med/high", "border", "stars" -> (gid, count)
        self._gl_ok  = False  # GL context initialised
        self._arrays = None   # set by set_arrays() from BuildThread
        self._lod    = "med"

        # Render timer
        self._t = QTimer(self)
        self._t.timeout.connect(self.update)
        self._t.start(16)

        self.setCursor(Qt.OpenHandCursor)
        self.setMinimumSize(200,200)

    def set_arrays(self, arrays):
        """Called from main thread when BuildThread finishes."""
        self._arrays = arrays
        if self._gl_ok:
            self._upload()   # GL already ready → upload now

    def _upload(self):
        """Upload numpy arrays to GPU. Must be called with GL context current."""
        arr = self._arrays
        if arr is None: return
        self.makeCurrent()

        for key,data in arr["dots"].items():
            gid = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, gid)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.tobytes(), GL_STATIC_DRAW)
            self._vbos[f"dots_{key}"] = (gid, len(data))

        gid = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, gid)
        glBufferData(GL_ARRAY_BUFFER, arr["border"].nbytes, arr["border"].tobytes(), GL_STATIC_DRAW)
        self._vbos["border"] = (gid, len(arr["border"]))

        gid = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, gid)
        glBufferData(GL_ARRAY_BUFFER, arr["stars"].nbytes, arr["stars"].tobytes(), GL_STATIC_DRAW)
        self._vbos["stars"] = (gid, len(arr["stars"]))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.doneCurrent()
        self.glReady.emit()

    # ── GL lifecycle ──────────────────────────────────────────
    def initializeGL(self):
        glClearColor(1/255,4/255,9/255,1.0)
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
        glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        self._gl_ok = True
        if self._arrays is not None:   # arrays already ready → upload
            self._upload()

    def resizeGL(self, w, h):
        glViewport(0,0,w,max(h,1))
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45.0, w/max(h,1), 0.05, 200.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if not self._vbos:
            return   # still loading

        glLoadIdentity()
        gluLookAt(0,0,self.cam_z, 0,0,0, 0,1,0)
        t = time.time()

        # auto-rotate + momentum
        if not self.dragging:
            if   self.rotation_dir=="west-east": self.rot_y += 0.004
            elif self.rotation_dir=="east-west": self.rot_y -= 0.004
            self.vel_x*=0.93; self.vel_y*=0.93
            self.rot_y+=self.vel_y; self.rot_x+=self.vel_x
            self.rot_x=max(-89,min(89,self.rot_x))

        if self.moon_enabled: self.moon_angle += 0.004

        # stars
        glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)
        if self.bg_affection: self.star_rot_x=self.rot_x; self.star_rot_y=self.rot_y
        else: self.star_rot_x*=0.96; self.star_rot_y*=0.96
        self._paint_stars()
        glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE)

        if self.sun_enabled: self._paint_sun()

        glPushMatrix()
        glRotatef(self.rot_x,1,0,0)
        glRotatef(math.degrees(self.rot_y),0,1,0)
        self._paint_atmo()
        self._paint_dots()
        self._paint_borders()
        self._paint_night()
        self._paint_marker(t)
        glPopMatrix()

        if self.moon_enabled: self._paint_moon()

    # ── Draw methods ──────────────────────────────────────────
    def _vbo_points(self, key):
        if key not in self._vbos: return
        gid,cnt = self._vbos[key]
        stride = 24   # 6 floats × 4 bytes
        glBindBuffer(GL_ARRAY_BUFFER, gid)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3,GL_FLOAT,stride, ctypes.c_void_p(0))
        glColorPointer (3,GL_FLOAT,stride, ctypes.c_void_p(12))
        glDrawArrays(GL_POINTS,0,cnt)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER,0)

    def _paint_stars(self):
        glPushMatrix()
        glRotatef(self.star_rot_x,1,0,0)
        glRotatef(math.degrees(self.star_rot_y),0,1,0)
        glPointSize(1.6); glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        self._vbo_points("stars")
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glPopMatrix()

    def _paint_atmo(self):
        glDepthMask(GL_FALSE); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        for rad,r,g,b,a in [(R*1.04,0,0.267,0.80,0.07),(R*1.12,0,0.133,0.67,0.03),(R*1.25,0,0.067,0.33,0.015)]:
            glColor4f(r,g,b,a); self._sphere(rad,32)
        glDepthMask(GL_TRUE)

    def _paint_dots(self):
        lod = "high" if self.cam_z<1.8 else ("med" if self.cam_z<3.2 else "low")
        self._lod = lod
        glPointSize({"low":4.5,"med":3.2,"high":2.0}[lod])
        glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        self._vbo_points(f"dots_{lod}")
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

    def _paint_borders(self):
        if "border" not in self._vbos: return
        gid,cnt = self._vbos["border"]
        if cnt<2: return
        glLineWidth(0.9); glColor4f(0,1.0,0.87,0.32)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glBindBuffer(GL_ARRAY_BUFFER,gid)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,None)
        glDrawArrays(GL_LINES,0,cnt)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER,0)

    def _paint_night(self):
        if not self.sun_enabled: return
        sx,sy,sz = 11.0,3.0,5.0; sl=math.sqrt(sx*sx+sy*sy+sz*sz)
        sx/=sl; sy/=sl; sz/=sl
        ry=-self.rot_y; rx=-math.radians(self.rot_x)
        cY,sY=math.cos(ry),math.sin(ry)
        lx=sx*cY+sz*sY; lz=-sx*sY+sz*cY; ly=sy
        cX,sX=math.cos(rx),math.sin(rx)
        lly=ly*cX-lz*sX; llz=ly*sX+lz*cX; llx=lx
        glDepthMask(GL_FALSE); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        N=48
        for i in range(N):
            lat0=math.pi*(-0.5+i/N); lat1=math.pi*(-0.5+(i+1)/N)
            z0,zr0=math.sin(lat0)*R,math.cos(lat0)*R
            z1,zr1=math.sin(lat1)*R,math.cos(lat1)*R
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(N+1):
                ang=2*math.pi*j/N; c,s=math.cos(ang),math.sin(ang)
                for z,zr in [(z0,zr0),(z1,zr1)]:
                    px,py,pz=c*zr,s*zr,z
                    dot=px*llx+py*lly+(pz/R)*llz
                    dark=max(0.0,min(0.75,-dot*1.5))
                    glColor4f(0,0.01,0.04,dark); glVertex3f(px,py,pz)
            glEnd()
        glDepthMask(GL_TRUE)

    def _paint_sun(self):
        glDepthMask(GL_FALSE); glDisable(GL_DEPTH_TEST); glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        glPushMatrix(); glTranslatef(5.5,1.5,2.5)
        for rad,r,g,b,a in [(0.12,1,1,0.87,0.9),(0.22,1,0.88,0.56,0.3),(0.38,1,0.75,0.27,0.12),(0.60,1,0.6,0.1,0.05)]:
            glColor4f(r,g,b,a)
            glBegin(GL_TRIANGLE_FAN); glVertex3f(0,0,0)
            for k in range(33):
                a_=2*math.pi*k/32; glVertex3f(math.cos(a_)*rad,math.sin(a_)*rad,0)
            glEnd()
        glPopMatrix(); glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_TRUE)

    def _paint_moon(self):
        mr=2.0; ma=self.moon_angle
        mx,my,mz=math.cos(ma)*mr,math.sin(ma*0.2)*0.3,math.sin(ma)*mr
        glPushMatrix(); glTranslatef(mx,my,mz)
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0,GL_POSITION,[11,3,5,0]); glLightfv(GL_LIGHT0,GL_DIFFUSE,[1,1,0.9,1])
        glLightfv(GL_LIGHT0,GL_AMBIENT,[0.05,0.05,0.1,1]); glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,[0.6,0.6,0.67,1])
        glColor4f(0.7,0.7,0.8,1); self._sphere(0.06,20)
        glDisable(GL_LIGHTING); glDisable(GL_LIGHT0)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE); glColor4f(0.67,0.73,0.8,0.06); self._sphere(0.10,16)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glPopMatrix()

    def _paint_marker(self, t):
        if self.marker_lat is None: return
        mx,my,mz=ll2xyz(self.marker_lat,self.marker_lng,R+0.02)
        cosY=math.cos(self.rot_y); sinY=math.sin(self.rot_y)
        mxr=mx*cosY-mz*sinY; mzr=mx*sinY+mz*cosY
        cosX=math.cos(math.radians(self.rot_x)); sinX=math.sin(math.radians(self.rot_x))
        mzr2=-my*sinX+mzr*cosX
        if mzr2<0.05: return
        p=math.sin(t*2.5); p2=math.sin(t*1.75+1)
        mc=(0.0,1.0,0.906) if self.marker_source=="gps" else (1.0,0.765,0.0)
        glPushMatrix(); glTranslatef(mx,my,mz)
        nx,ny,nz=_norm((mx,my,mz))
        ang=math.degrees(math.acos(max(-1,min(1,nz)))); ax,ay=-ny,nx; al=math.sqrt(ax*ax+ay*ay)
        if al>1e-6: glRotatef(ang,ax/al,ay/al,0)
        glDepthMask(GL_FALSE); glBlendFunc(GL_SRC_ALPHA,GL_ONE)
        glColor3f(*mc); glPointSize(9)
        glBegin(GL_POINTS); glVertex3f(0,0,0); glEnd()
        s1=1+0.45*p;  a1=0.5+0.35*abs(p);  glColor4f(*mc,a1);  self._ring(0.028*s1,0.036*s1,48)
        s2=1+0.3*p2;  a2=0.2+0.2*abs(p2);  glColor4f(*mc,a2);  self._ring(0.050*s2,0.060*s2,48)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_TRUE); glPopMatrix()

    # ── Primitives ────────────────────────────────────────────
    def _sphere(self, rad, sl):
        st=sl//2
        for i in range(st):
            lat0=math.pi*(-0.5+i/st); lat1=math.pi*(-0.5+(i+1)/st)
            z0,zr0=math.sin(lat0)*rad,math.cos(lat0)*rad
            z1,zr1=math.sin(lat1)*rad,math.cos(lat1)*rad
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(sl+1):
                lng=2*math.pi*j/sl; c,s=math.cos(lng),math.sin(lng)
                glVertex3f(c*zr0,s*zr0,z0); glVertex3f(c*zr1,s*zr1,z1)
            glEnd()

    def _ring(self, ri, ro, n):
        glBegin(GL_TRIANGLE_STRIP)
        for k in range(n+1):
            a=2*math.pi*k/n; c,s=math.cos(a),math.sin(a)
            glVertex3f(c*ri,s*ri,0); glVertex3f(c*ro,s*ro,0)
        glEnd()

    # ── Input ─────────────────────────────────────────────────
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            self.dragging=True; self.last_pos=e.pos(); self.vel_x=self.vel_y=0
            self.setCursor(Qt.ClosedHandCursor)
    def mouseMoveEvent(self,e):
        if self.dragging:
            dx=e.x()-self.last_pos.x(); dy=e.y()-self.last_pos.y()
            self.rot_y+=dx*0.005; self.rot_x=max(-89,min(89,self.rot_x+dy*0.3))
            self.vel_y=dx*0.005; self.vel_x=dy*0.3; self.last_pos=e.pos()
    def mouseReleaseEvent(self,e):
        if e.button()==Qt.LeftButton: self.dragging=False; self.setCursor(Qt.OpenHandCursor)
    def wheelEvent(self,e):
        self.cam_z=max(1.3,min(5.5,self.cam_z-e.angleDelta().y()/120*0.15))

    # ── API ───────────────────────────────────────────────────
    def set_location(self,lat,lng,src="ip"):
        self.marker_lat=lat; self.marker_lng=lng; self.marker_source=src
        self.rot_y=-(math.radians(lng+180))+math.pi; self.rot_x=-lat*0.5
    def get_stats(self):
        lod=self._lod; cnt=self._vbos.get(f"dots_{lod}",(None,0))[1]
        border_cnt=self._vbos.get("border",(None,0))[1]
        pct={"low":25,"med":55,"high":90}.get(lod,55)
        return lod.upper(),cnt,pct,round(self.cam_z,2),border_cnt


# ═══════════════════════════════════════════════════════════════
# SMALL UI HELPERS
# ═══════════════════════════════════════════════════════════════
def mono(sz=9):
    f=QFont("Courier New",sz); f.setStyleHint(QFont.Monospace); return f
def sans(sz=10,bold=False):
    f=QFont("Arial",sz)
    if bold: f.setBold(True)
    return f

def lbl(text,color=None,sz=9,mono_=False,parent=None):
    w=QLabel(text,parent)
    w.setFont(mono(sz) if mono_ else sans(sz))
    w.setStyleSheet(f"color:{_c(color or C_TEXT)};background:transparent;")
    return w

class HexFrame(QFrame):
    def __init__(self,p=None):
        super().__init__(p); self.setStyleSheet(f"QFrame{{{PANEL_CSS}}}")

class Blinker(QLabel):
    def __init__(self,p=None):
        super().__init__("●",p); self._on=True; self.setFont(mono(8))
        t=QTimer(self); t.timeout.connect(self._flip); t.start(700); self._flip()
    def _flip(self):
        self._on=not self._on; a=255 if self._on else 60
        self.setStyleSheet(f"color:rgba(0,255,231,{a});background:transparent;")


# ═══════════════════════════════════════════════════════════════
# PANELS
# ═══════════════════════════════════════════════════════════════
class LocPanel(HexFrame):
    def __init__(self,p=None):
        super().__init__(p)
        v=QVBoxLayout(self); v.setContentsMargins(10,8,10,8); v.setSpacing(3)
        h=QHBoxLayout(); h.addWidget(lbl("YOUR LOCATION",C_DIM,7,True)); h.addStretch()
        self._badge=QLabel("⬤ IP"); self._badge.setFont(mono(6))
        self._badge.setStyleSheet(f"color:{_c(C_ACCENT3)};border:1px solid {_c(C_DIM)};padding:1px 4px;background:rgba(255,195,0,18);")
        h.addWidget(self._badge); v.addLayout(h)
        def row(k):
            hh=QHBoxLayout(); hh.addWidget(lbl(k,C_DIM,8)); hh.addStretch()
            val=lbl("--",C_ACCENT3,8,True); hh.addWidget(val); v.addLayout(hh); return val
        self._ip=row("IP"); self._ip.setStyleSheet(f"color:{_c(C_ACCENT)};background:transparent;font-size:8pt;")
        self._isp=row("ISP")
        for _ in range(1):
            sp=QFrame(); sp.setFrameShape(QFrame.HLine); sp.setStyleSheet(f"color:{_c(C_BORDER)};"); v.addWidget(sp)
        self._city=row("City"); self._region=row("Region"); self._country=row("Country")
        self._lat=row("Lat"); self._lng=row("Lng")
        sp2=QFrame(); sp2.setFrameShape(QFrame.HLine); sp2.setStyleSheet(f"color:{_c(C_BORDER)};"); v.addWidget(sp2)
        sr=QHBoxLayout(); sr.addWidget(Blinker()); sr.addSpacing(4)
        self._status=lbl("SIGNAL LOCKED",C_ACCENT,7,True); sr.addWidget(self._status); sr.addStretch(); v.addLayout(sr)
    def apply(self,d,src):
        self._ip.setText(d.get("ip","--") or "--")
        isp=d.get("isp","--") or "--"; self._isp.setText(isp[:22]+"…" if len(isp)>22 else isp)
        self._city.setText(d.get("city","--") or "--"); self._region.setText(d.get("region","--") or "--")
        self._country.setText(d.get("country","--") or "--")
        self._lat.setText(f"{d.get('lat',0):.4f}°"); self._lng.setText(f"{d.get('lng',0):.4f}°")
        if src=="gps":
            self._badge.setText("⬤ GPS"); self._badge.setStyleSheet(f"color:{_c(C_ACCENT)};border:1px solid {_c(C_ACCENT)};padding:1px 4px;background:rgba(0,255,231,18);")
            self._status.setText("GPS LOCKED")
        else:
            self._badge.setText("⬤ IP"); self._badge.setStyleSheet(f"color:{_c(C_ACCENT3)};border:1px solid {_c(C_DIM)};padding:1px 4px;background:rgba(255,195,0,18);")
            self._status.setText("IP LOCATED")

class StatsPanel(HexFrame):
    def __init__(self,p=None):
        super().__init__(p)
        v=QVBoxLayout(self); v.setContentsMargins(10,8,10,8); v.setSpacing(3)
        v.addWidget(lbl("GLOBE STATS",C_DIM,7,True))
        def row(k):
            h=QHBoxLayout(); h.addWidget(lbl(k,C_DIM,8)); h.addStretch()
            val=lbl("--",C_ACCENT,8,True); h.addWidget(val); v.addLayout(h); return val
        self._d=row("Dots"); self._b=row("Borders"); self._f=row("FPS"); self._l=row("LOD"); self._z=row("Zoom")
    def update(self,dots,borders,fps,lod,zoom):
        self._d.setText(f"{dots:,}"); self._b.setText(f"{borders:,}")
        self._f.setText(str(fps)); self._l.setText(lod); self._z.setText(str(zoom))

class LodPanel(HexFrame):
    def __init__(self,p=None):
        super().__init__(p); self.setMinimumWidth(150)
        v=QVBoxLayout(self); v.setContentsMargins(10,8,10,8); v.setSpacing(2)
        v.addWidget(lbl("RENDER DENSITY",C_DIM,7,True))
        self._zv=QLabel("--"); self._zv.setFont(mono(22))
        self._zv.setStyleSheet(f"color:{_c(C_ACCENT)};background:transparent;font-weight:900;"); v.addWidget(self._zv)
        self._ll=lbl("MED",C_ACCENT,8,True); v.addWidget(self._ll)
        self._bg=QFrame(); self._bg.setFixedHeight(3); self._bg.setStyleSheet(f"background:{_c(C_BORDER)};border:none;"); v.addWidget(self._bg)
        self._bar=QFrame(self._bg); self._bar.setGeometry(0,0,80,3); self._bar.setStyleSheet(f"background:{_c(C_ACCENT)};border:none;")
    def resizeEvent(self,e): super().resizeEvent(e); self._bg.setFixedWidth(max(10,self.width()-20))
    def update(self,lod,zoom,pct):
        self._zv.setText(str(zoom)); self._ll.setText(lod)
        self._bar.setGeometry(0,0,max(2,int(self._bg.width()*pct/100)),3)

class CtrlPanel(HexFrame):
    def __init__(self,p=None):
        super().__init__(p)
        v=QVBoxLayout(self); v.setContentsMargins(10,8,10,8); v.setSpacing(3)
        v.addWidget(lbl("CONTROLS",C_DIM,7,True))
        for key,desc in [("DRAG","Rotate globe"),("SCROLL","Zoom in/out"),("WHEEL","Zoom in/out")]:
            h=QHBoxLayout(); k=QLabel(key); k.setFont(mono(7))
            k.setStyleSheet(f"color:{_c(C_ACCENT)};border:1px solid {_c(C_DIM)};padding:1px 5px;background:transparent;"); k.setFixedWidth(55)
            h.addWidget(k); h.addWidget(lbl(desc,C_DIM,8)); h.addStretch(); v.addLayout(h)

class SettingsPanel(QFrame):
    rotDir=pyqtSignal(str); sunSig=pyqtSignal(bool); moonSig=pyqtSignal(bool); bgSig=pyqtSignal(bool)
    def __init__(self,p=None):
        super().__init__(p); self.setFixedWidth(268); self.setStyleSheet(f"QFrame{{{PANEL_CSS}}}")
        v=QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
        hdr=QLabel("  ⬤  SETTINGS"); hdr.setFont(mono(8))
        hdr.setStyleSheet(f"color:{_c(C_ACCENT)};background:rgba(0,255,231,12);padding:8px 10px;border-bottom:1px solid {_c(C_BORDER)};")
        v.addWidget(hdr)
        def sec():
            w=QWidget(); wv=QVBoxLayout(w); wv.setContentsMargins(10,8,10,8); wv.setSpacing(5)
            sp=QFrame(); sp.setFrameShape(QFrame.HLine); sp.setStyleSheet(f"color:{_c(C_BORDER)};")
            return w,wv,sp
        w1,s1,sp1=sec(); s1.addWidget(lbl("ROTATION DIRECTION",C_DIM,6,True))
        rr=QHBoxLayout(); self._rew=QPushButton("◀ E→W"); self._rwe=QPushButton("W→E ▶"); self._rst=QPushButton("■ STOP")
        for b in (self._rew,self._rwe,self._rst): b.setStyleSheet(BTN_BASE); rr.addWidget(b)
        s1.addLayout(rr); v.addWidget(w1); v.addWidget(sp1)
        self._rew.clicked.connect(lambda:self._rot("east-west")); self._rwe.clicked.connect(lambda:self._rot("west-east")); self._rst.clicked.connect(lambda:self._rot("stopped"))
        self._rot("west-east")
        w2,s2,sp2=sec()
        self._son=QPushButton("ON"); self._sof=QPushButton("OFF"); self._mon=QPushButton("ON"); self._mof=QPushButton("OFF")
        for label,on_b,off_b,fn in [("☀  Sun",self._son,self._sof,self._st),("🌙 Moon",self._mon,self._mof,self._mt)]:
            hh=QHBoxLayout(); hh.addWidget(lbl(label,C_TEXT,9)); hh.addStretch(); hh.addWidget(on_b); hh.addWidget(off_b); s2.addLayout(hh)
            on_b.clicked.connect(lambda _,f=fn:f(True)); off_b.clicked.connect(lambda _,f=fn:f(False))
        v.addWidget(w2); v.addWidget(sp2); self._st(True); self._mt(False)
        w3,s3,_=sec(); s3.addWidget(lbl("BACKGROUND DRAG EFFECT",C_DIM,6,True))
        s3.addWidget(lbl("Stars move with globe when dragged",C_DIM,7))
        br=QHBoxLayout(); self._bon=QPushButton("ENABLE"); self._bof=QPushButton("DISABLE")
        for b in (self._bon,self._bof): b.setStyleSheet(BTN_BASE); br.addWidget(b)
        s3.addLayout(br); v.addWidget(w3); v.addStretch()
        self._bon.clicked.connect(lambda:self._bt(True)); self._bof.clicked.connect(lambda:self._bt(False)); self._bt(True)
    def _rot(self,d):
        self._rew.setStyleSheet(togbtn(d=="east-west")); self._rwe.setStyleSheet(togbtn(d=="west-east"))
        self._rst.setStyleSheet(togbtn(d=="stopped",C_ACCENT2) if d=="stopped" else BTN_BASE); self.rotDir.emit(d)
    def _st(self,v): self._son.setStyleSheet(togbtn(v)); self._sof.setStyleSheet(togbtn(not v,C_ACCENT2)); self.sunSig.emit(v)
    def _mt(self,v): self._mon.setStyleSheet(togbtn(v)); self._mof.setStyleSheet(togbtn(not v,C_ACCENT2)); self.moonSig.emit(v)
    def _bt(self,v): self._bon.setStyleSheet(togbtn(v)); self._bof.setStyleSheet(togbtn(not v,C_ACCENT2)); self.bgSig.emit(v)


# ═══════════════════════════════════════════════════════════════
# TOAST
# ═══════════════════════════════════════════════════════════════
class ToastCard(QFrame):
    _C={"error":C_ACCENT2,"warn":C_ACCENT3,"info":C_ACCENT}
    def __init__(self,kind,tag,msg,p=None):
        super().__init__(p); c=self._C.get(kind,C_ACCENT)
        self.setStyleSheet(f"QFrame{{background:rgba(5,10,24,242);border:1px solid {_c(c)};border-radius:2px;}}")
        v=QVBoxLayout(self); v.setContentsMargins(8,5,8,5); v.setSpacing(2)
        h=QHBoxLayout()
        tg=QLabel(tag); tg.setFont(mono(7)); tg.setStyleSheet(f"color:{_c(c)};background:transparent;")
        tm=QLabel(datetime.now().strftime("%H:%M:%S")); tm.setFont(mono(6)); tm.setStyleSheet(f"color:{_c(C_DIM)};background:transparent;")
        h.addWidget(tg); h.addStretch(); h.addWidget(tm); v.addLayout(h)
        ml=QLabel(msg); ml.setFont(sans(8)); ml.setWordWrap(True); ml.setStyleSheet(f"color:{_c(C_TEXT)};background:transparent;")
        v.addWidget(ml); self.setMaximumWidth(290)

class ToastManager(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setAttribute(Qt.WA_TranslucentBackground)
        v=QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(4); v.addStretch(); self._cards=[]
    def toast(self,kind,tag,msg,ms=0):
        c=ToastCard(kind,tag,msg,self); self.layout().addWidget(c); c.show(); self._cards.append(c)
        if ms>0: QTimer.singleShot(ms,lambda cc=c:self._die(cc))
        self.adjustSize()
    def _die(self,c):
        if c in self._cards: self._cards.remove(c); c.deleteLater(); self.adjustSize()
    def clear(self):
        for c in list(self._cards): self._die(c)


# ═══════════════════════════════════════════════════════════════
# LOADING OVERLAY  — sits on top of the GL widget
# ═══════════════════════════════════════════════════════════════
class LoadOverlay(QWidget):
    """
    Plain QWidget painted over the GlobeWidget.
    Fades out and hides itself when done.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet("background:rgb(1,4,9);")

        v=QVBoxLayout(self); v.setAlignment(Qt.AlignCenter); v.setSpacing(14)
        tl=QLabel("AAYAN SYSTEMS — GLOBE v1.0"); tl.setFont(mono(9))
        tl.setStyleSheet(f"color:{_c(C_ACCENT)};letter-spacing:5px;background:transparent;"); tl.setAlignment(Qt.AlignCenter); v.addWidget(tl)

        self._pct=QLabel("0%"); self._pct.setFont(mono(28))
        self._pct.setStyleSheet(f"color:{_c(C_ACCENT)};font-weight:900;background:transparent;"); self._pct.setAlignment(Qt.AlignCenter); v.addWidget(self._pct)

        bb=QFrame(); bb.setFixedSize(320,3); bb.setStyleSheet("background:rgba(0,255,231,20);border:none;"); v.addWidget(bb,0,Qt.AlignCenter)
        self._bar=QFrame(bb); self._bar.setGeometry(0,0,0,3); self._bar.setStyleSheet(f"background:{_c(C_ACCENT)};border:none;")

        self._msg=QLabel("Building globe data…"); self._msg.setFont(mono(7))
        self._msg.setStyleSheet(f"color:{_c(C_DIM)};letter-spacing:2px;background:transparent;"); self._msg.setAlignment(Qt.AlignCenter); v.addWidget(self._msg)

        self._fade_timer=QTimer(self); self._fade_timer.timeout.connect(self._fade_step)
        self._opacity=1.0

    def set_progress(self, pct, msg):
        self._pct.setText(f"{pct}%")
        self._msg.setText(msg)
        self._bar.setGeometry(0,0,int(320*pct/100),3)

    def fade_out(self):
        self._fade_timer.start(16)

    def _fade_step(self):
        self._opacity=max(0.0,self._opacity-0.05)
        self.setWindowOpacity(self._opacity)
        # Use stylesheet alpha instead (works for child widget)
        a=int(self._opacity*255)
        self.setStyleSheet(f"background:rgba(1,4,9,{a});")
        if self._opacity<=0:
            self._fade_timer.stop()
            self.hide()


# ═══════════════════════════════════════════════════════════════
# HEADER  +  BRACKETS  +  FLOAT BUTTONS
# ═══════════════════════════════════════════════════════════════
class Header(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setAttribute(Qt.WA_StyledBackground); self.setFixedHeight(52)
        self.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 rgba(1,4,9,242),stop:1 rgba(1,4,9,0));border-bottom:1px solid rgba(0,255,231,15);")
        h=QHBoxLayout(self); h.setContentsMargins(14,0,14,0)
        left=QVBoxLayout()
        name=QLabel(); name.setFont(mono(11)); name.setStyleSheet("background:transparent;")
        name.setText('<span style="color:rgb(0,255,231)">Aayan</span> <span style="color:white">Globe</span>')
        ver=QLabel("VERSION 1.0 — DOT FIELD ENGINE"); ver.setFont(mono(5))
        ver.setStyleSheet(f"color:{_c(C_ACCENT3)};letter-spacing:4px;background:transparent;")
        left.addWidget(name); left.addWidget(ver); h.addLayout(left); h.addStretch()
        mid=QHBoxLayout(); mid.setSpacing(5); mid.addWidget(Blinker())
        ll=QLabel("LIVE TRACKING"); ll.setFont(mono(7)); ll.setStyleSheet(f"color:{_c(C_ACCENT)};letter-spacing:2px;background:transparent;")
        mid.addWidget(ll); h.addLayout(mid); h.addStretch()
        self._clk=QLabel("--:--:--"); self._clk.setFont(mono(8)); self._clk.setStyleSheet(f"color:{_c(C_DIM)};letter-spacing:2px;background:transparent;")
        h.addWidget(self._clk)
        t=QTimer(self); t.timeout.connect(self._tick); t.start(1000); self._tick()
    def _tick(self): self._clk.setText(datetime.now().strftime("%H:%M:%S"))

class Brackets(QWidget):
    def __init__(self,p=None):
        super().__init__(p); self.setAttribute(Qt.WA_TransparentForMouseEvents); self.setAttribute(Qt.WA_NoSystemBackground)
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(QPen(QColor(0,255,231,77),2))
        w,h=self.width(),self.height(); sz=28; top=54
        p.drawLine(0,top,0,top+sz); p.drawLine(0,top,sz,top)
        p.drawLine(w,top,w,top+sz); p.drawLine(w,top,w-sz,top)
        p.drawLine(0,h,0,h-sz); p.drawLine(0,h,sz,h)
        p.drawLine(w,h,w,h-sz); p.drawLine(w,h,w-sz,h)

class FloatBtn(QPushButton):
    def __init__(self,txt,p=None):
        super().__init__(txt,p); self.setFixedSize(38,38); self._a=False; self._rs()
    def set_active(self,v): self._a=v; self._rs()
    def _rs(self):
        if self._a: self.setStyleSheet(f"QPushButton{{background:rgba(0,255,231,30);border:1px solid {_c(C_ACCENT)};color:{_c(C_ACCENT)};font-size:18px;}}")
        else: self.setStyleSheet(f"QPushButton{{background:rgba(5,12,28,235);border:1px solid {_c(C_BORDER)};color:{_c(C_ACCENT)};font-size:18px;}}QPushButton:hover{{background:rgba(0,255,231,18);}}")


# ═══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aayan Globe  v1.0")
        self.resize(1280,800); self.setMinimumSize(900,600)
        self.setStyleSheet("QMainWindow{background:rgb(1,4,9);}")
        self._hud_vis=True; self._set_vis=False; self._fps_t=[]

        # Central container
        self._root=QWidget(); self._root.setStyleSheet("background:transparent;")
        self.setCentralWidget(self._root)

        # ── Globe widget (always visible, always rendering) ──
        self._globe=GlobeWidget(self._root)

        # ── HUD overlay ──
        self._hud=QWidget(self._root); self._hud.setAttribute(Qt.WA_TranslucentBackground)
        self._header=Header(self._hud)
        self._brackets=Brackets(self._hud)
        self._lod_p=LodPanel(self._hud)
        self._ctrl_p=CtrlPanel(self._hud)
        self._stat_p=StatsPanel(self._hud)
        self._loc_p=LocPanel(self._hud)

        # Float buttons (on root, above everything)
        self._ham=FloatBtn("≡",self._root); self._setbtn=FloatBtn("⚙",self._root)
        self._settings=SettingsPanel(self._root); self._settings.hide()
        self._toasts=ToastManager(self._root)

        # ── LOADING OVERLAY (on top of everything) ──
        self._overlay=LoadOverlay(self._root)

        # ── Build thread (starts immediately) ──
        self._builder=BuildThread()
        self._builder.progress.connect(self._overlay.set_progress)
        self._builder.done.connect(self._on_arrays_ready)
        self._builder.start()

        # ── Globe ready signal ──
        self._globe.glReady.connect(self._on_gl_ready)

        # Settings wiring
        self._settings.rotDir.connect(lambda d: setattr(self._globe,'rotation_dir',d))
        self._settings.sunSig.connect(lambda v: setattr(self._globe,'sun_enabled',v))
        self._settings.moonSig.connect(lambda v: setattr(self._globe,'moon_enabled',v))
        self._settings.bgSig.connect(lambda v: setattr(self._globe,'bg_affection',v))
        self._ham.clicked.connect(self._toggle_hud)
        self._setbtn.clicked.connect(self._toggle_settings)

        # Geo
        self._geo=GeoThread(); self._geo.result.connect(self._on_geo)
        self._geo.message.connect(lambda k,t,m:self._toasts.toast(k,t,m,3000))
        self._geo.start()

        # Stats refresh
        self._ui_t=QTimer(self); self._ui_t.timeout.connect(self._refresh); self._ui_t.start(500)

        self._layout()

    def _layout(self):
        W,H=self.width(),self.height()
        self._globe.setGeometry(0,0,W,H)
        self._hud.setGeometry(0,0,W,H)
        self._header.setGeometry(0,0,W,52)
        self._brackets.setGeometry(0,0,W,H)
        self._lod_p.setGeometry(14,62,160,115)
        self._ctrl_p.setGeometry(W-215,62,200,108)
        self._stat_p.setGeometry(W-170,H-178,155,163)
        self._loc_p.setGeometry(14,H-248,245,232)
        self._ham.setGeometry(W-50,8,38,38)
        self._setbtn.setGeometry(W-50,50,38,38)
        self._settings.setGeometry(W-322,8,268,370)
        self._toasts.setGeometry(W-308,H-420,296,400)
        self._overlay.setGeometry(0,0,W,H)   # full-screen overlay

    def resizeEvent(self,e):
        super().resizeEvent(e); self._layout()

    def _on_arrays_ready(self, arrays):
        """Called on main thread when BuildThread finishes. Pass arrays to globe."""
        self._globe.set_arrays(arrays)
        # If glReady already fired, fade out now. Otherwise wait for glReady.
        if self._globe._vbos:
            self._overlay.set_progress(100,"Ready!")
            QTimer.singleShot(400, self._overlay.fade_out)

    def _on_gl_ready(self):
        """Called when GPU upload is complete."""
        self._overlay.set_progress(100,"Ready!")
        QTimer.singleShot(400, self._overlay.fade_out)

    def _on_geo(self,d):
        src=d.get("source","ip"); self._loc_p.apply(d,src)
        self._globe.set_location(d.get("lat",23.35),d.get("lng",85.33),src)
        self._toasts.clear()
        self._toasts.toast("info","LOCATION LOCKED",f"{d.get('ip','?')} · {d.get('city','?')}, {d.get('country','?')}",5000)

    def _toggle_hud(self):
        self._hud_vis=not self._hud_vis; self._ham.set_active(not self._hud_vis); self._hud.setVisible(self._hud_vis)
    def _toggle_settings(self):
        self._set_vis=not self._set_vis; self._setbtn.set_active(self._set_vis); self._settings.setVisible(self._set_vis)

    def _refresh(self):
        now=time.time(); self._fps_t.append(now); self._fps_t=[t for t in self._fps_t if now-t<2]
        fps=len(self._fps_t)
        if self._globe._vbos:
            lod,cnt,pct,zoom,borders=self._globe.get_stats()
            self._lod_p.update(lod,zoom,pct); self._stat_p.update(cnt,borders,fps,lod,zoom)

    def closeEvent(self,e):
        self._builder.quit(); self._builder.wait(1000)
        self._geo.quit(); self._geo.wait(800); super().closeEvent(e)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    fmt=QSurfaceFormat(); fmt.setSamples(4); fmt.setDepthBufferSize(24)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer); QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts,True)

    app=QApplication(sys.argv); app.setApplicationName("Aayan Globe")
    pal=QPalette()
    for role,col in [(QPalette.Window,C_BG),(QPalette.WindowText,C_TEXT),(QPalette.Base,C_PANEL),
                     (QPalette.AlternateBase,C_BG),(QPalette.Text,C_TEXT),(QPalette.Button,C_PANEL),
                     (QPalette.ButtonText,C_ACCENT),(QPalette.Highlight,C_ACCENT),(QPalette.HighlightedText,C_BG)]:
        pal.setColor(role,col)
    app.setPalette(pal)

    win=MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()