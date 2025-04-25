#-------------------------------
# imports
#-------------------------------

# builtins
import os,sys,time,traceback
from math import hypot

# must be installed using pip
# python3 -m pip install opencv-python
import numpy as np
import cv2

# local clayton libs
# Pastikan file frame_capture.py dan frame_draw.py ada di direktori yang sama
# atau dapat diimpor oleh Python
try:
    import frame_capture
    import frame_draw
except ImportError:
    print("ERROR: Pastikan file frame_capture.py dan frame_draw.py ada.")
    print("Anda bisa mendapatkannya dari repositori asli CamRuler.")
    sys.exit(1)


#-------------------------------
# default settings
#-------------------------------

# camera values
camera_id = 0
camera_width = 640
camera_height = 480
camera_frame_rate = 30
#camera_fourcc = cv2.VideoWriter_fourcc(*"YUYV")
camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# auto measure mouse events
auto_percent = 0.2
auto_threshold = 127
auto_blur = 5

# normalization mouse events
norm_alpha = 0
norm_beta = 255

#-------------------------------
# read config file
#-------------------------------

# you can make a config file "camruler_config.csv"
# this is a comma-separated file with one "item,value" pair per line
# you can also use a "=" separated pair like "item=value"
# you can use # to comment a line
# the items must be named like the default variables above

# read local config values
configfile = 'camruler_config.csv'
if os.path.isfile(configfile):
    with open(configfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#' and (',' in line or '=' in line):
                if ',' in line:
                    item,value = [x.strip() for x in line.split(',',1)]
                elif '=' in line:
                    item,value = [x.strip() for x in line.split('=',1)]
                else:
                    continue
                if item in 'camera_id camera_width camera_height camera_frame_rate camera_fourcc auto_percent auto_threshold auto_blur norm_alpha norm_beta cal_range unit_suffix'.split(): # Tambahkan var lain jika perlu
                    try:
                        # Coba konversi ke tipe data yang sesuai
                        if item in ['camera_id', 'camera_width', 'camera_height', 'camera_frame_rate', 'auto_threshold', 'auto_blur', 'norm_alpha', 'norm_beta', 'cal_range']:
                            value = int(value)
                        elif item in ['auto_percent']:
                            value = float(value)
                        # Untuk camera_fourcc dan unit_suffix biarkan sebagai string
                        exec(f'{item}="{value}"' if isinstance(value, str) else f'{item}={value}')
                        print('CONFIG:',(item,value))
                    except Exception as e:
                        print(f'CONFIG ERROR reading item "{item}" with value "{value}": {e}')

#-------------------------------
# camera setup
#-------------------------------

# get camera id from argv[1]
# example "python3 camruler.py 2"
if len(sys.argv) > 1:
    arg_cam_id = sys.argv[1]
    if arg_cam_id.isdigit():
        camera_id = int(arg_cam_id)
    else:
        # Jika argumen bukan digit, coba gunakan sebagai path video
        if os.path.isfile(arg_cam_id):
            camera_id = arg_cam_id
            print(f"Using video file: {camera_id}")
        else:
            print(f"Invalid camera ID or file path: {arg_cam_id}. Using default: {camera_id}")


# camera thread setup
camera = frame_capture.Camera_Thread()
camera.camera_source = camera_id # SET THE CORRECT CAMERA NUMBER or path
camera.camera_width  = camera_width
camera.camera_height = camera_height
camera.camera_frame_rate = camera_frame_rate
camera.camera_fourcc = camera_fourcc

#1 start camera thread
camera.start()
# Beri waktu kamera untuk inisialisasi, terutama jika dari file
time.sleep(1.0)


# initial camera values (shortcuts for below)
width  = camera.camera_width
height = camera.camera_height
# --- FIX: Beri nama variabel area ---
area = width*height
cx = int(width/2)
cy = int(height/2)
dm = hypot(cx,cy) # max pixel distance
frate  = camera.camera_frame_rate
# --- FIX: Cetak variabel area ---
print('CAMERA:',[camera.camera_source,width,height,area,frate])
if area == 0:
    print("ERROR: Camera failed to initialize properly, width/height is zero.")
    camera.stop()
    sys.exit(1)


#-------------------------------
# frame drawing/text module
#-------------------------------

draw = frame_draw.DRAW()
draw.width = width
draw.height = height

#-------------------------------
# conversion (pixels to measure)
#-------------------------------

# distance units designator
unit_suffix = 'mm'

# calibrate every N pixels
pixel_base = 10

# maximum field of view from center to farthest edge
# ini yang akan diubah nantinya
cal_range = 27 # Default, bisa di-override config

# initial calibration values table {pixels:scale}
# this is based on the frame size and the cal_range
# Pastikan dm tidak nol
if dm == 0: dm = hypot(width, height) # Fallback jika cx/cy nol
cal = dict([(x,cal_range/dm) for x in range(0,int(dm)+1,pixel_base)])

# calibration loop values
# inside of main loop below
cal_base = 5
cal_last = None

# calibration update
def cal_update(x,y,unit_distance):
    global cal # Pastikan kita memodifikasi cal global

    # basics
    pixel_distance = hypot(x,y)
    if pixel_distance == 0:
        print("CAL WARN: Pixel distance is zero, cannot update calibration.")
        return # Hindari pembagian dengan nol

    scale = abs(unit_distance/pixel_distance)
    target = baseround(abs(pixel_distance),pixel_base)

    # low-high values in distance
    low  = target*scale - (cal_base/2)
    high = target*scale + (cal_base/2)

    # get low start point in pixels
    start = target
    if unit_distance <= cal_base:
        start = 0
    else:
        # Pastikan start tidak negatif
        while start*scale > low and start >= pixel_base:
            start -= pixel_base
        start = max(0, start) # Jaga agar tidak negatif

    # get high stop point in pixels
    stop = target
    max_pixel_dist = max(cal.keys()) if cal else int(dm)
    # Cek apakah cal_range sudah tercapai
    # Gunakan nilai dm jika cal kosong atau max_pixel_dist tidak valid
    effective_max_pixel_dist = max_pixel_dist if max_pixel_dist > 0 else int(dm)
    if unit_distance >= baseround(cal_range,pixel_base):
        stop = effective_max_pixel_dist
    else:
        while stop*scale < high and stop <= effective_max_pixel_dist:
            stop += pixel_base
        # Pastikan stop tidak melebihi batas maksimum piksel
        stop = min(stop, effective_max_pixel_dist)


    # set scale
    print(f'CAL INFO: Updating scale={scale:.4f} for pixels {start} to {stop}')
    for px in range(start,stop+1,pixel_base):
        # Pastikan px ada sebagai key sebelum update
        # Ini seharusnya tidak terjadi dengan logika di atas, tapi sebagai pengaman
        # if px in cal:
        cal[px] = scale
        # print(f'CAL: {px} {scale:.4f}') # Mungkin terlalu verbose
    print(f'CAL DONE: Updated pixels {start} to {stop}')


# read local calibration data
calfile = 'camruler_cal.csv'
if os.path.isfile(calfile):
    print(f"Loading calibration from: {calfile}")
    try:
        with open(calfile) as f:
            for line in f:
                line = line.strip()
                if line and line[0] in ('d',): # Hanya proses baris data 'd'
                    try:
                        axis,pixels_str,scale_str = [_.strip() for _ in line.split(',',2)]
                        if axis == 'd':
                            pixels = int(pixels_str)
                            scale = float(scale_str)
                            # Pastikan pixel ada dalam rentang yang mungkin
                            if 0 <= pixels <= int(dm) + pixel_base:
                                cal[pixels] = scale
                                print(f'LOAD: {pixels} {scale}')
                            else:
                                print(f'LOAD WARN: Pixel value {pixels} out of range, skipping.')
                    except ValueError as e:
                        print(f"LOAD ERROR: Invalid format in line: '{line}'. Error: {e}")
                    except Exception as e:
                         print(f"LOAD ERROR: Unexpected error processing line: '{line}'. Error: {e}")
    except Exception as e:
        print(f"ERROR: Could not read calibration file {calfile}. Error: {e}")


# convert pixels to units
def conv(x,y):

    d = distance(0,0,x,y)
    # Gunakan nilai scale terdekat jika d tidak tepat kelipatan pixel_base
    base_d = baseround(d,pixel_base)

    # Cari kunci terdekat jika base_d tidak ada (misal karena pembulatan)
    if base_d not in cal:
        # Cari kunci terdekat yang lebih kecil atau sama
        keys = sorted([k for k in cal.keys() if k <= base_d], reverse=True)
        if keys:
            base_d = keys[0]
        else:
            # Jika tidak ada yang lebih kecil, cari yang terdekat lebih besar
            keys = sorted([k for k in cal.keys() if k >= base_d])
            if keys:
                base_d = keys[0]
            else:
                # Jika cal kosong atau tidak ada kunci yang cocok, gunakan nilai default
                print(f"CONV WARN: No suitable calibration key found for pixel distance {d:.2f} (base {base_d}). Using default scale.")
                return x * (cal_range / dm), y * (cal_range / dm)

    scale = cal[base_d]
    if scale == 0:
        print(f"CONV WARN: Scale is zero for pixel distance {d:.2f} (base {base_d}). Check calibration.")
        return 0.0, 0.0 # Hindari hasil tak terduga

    return x*scale,y*scale

# round to a given base
def baseround(x,base=1):
    if base == 0: return int(x) # Hindari pembagian dengan nol
    return int(base * round(float(x)/base))

# distance formula 2D
def distance(x1,y1,x2,y2):
    return hypot(x1-x2,y1-y2)

#-------------------------------
# define frames
#-------------------------------

# define display frame
framename = "CamRuler ~ ClaytonDarwin's Youtube Channel"
cv2.namedWindow(framename,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)

#-------------------------------
# key events
#-------------------------------

key_last = 0
key_flags = {'config':False, # c key
             'auto':False,   # a key
             'thresh':False, # t key
             'percent':False,# p key
             'norms':False,  # n key
             'rotate':False, # r key
             'lock':False,   #
             }

def key_flags_clear():

    global key_flags

    for key in list(key_flags.keys()):
        if key not in ('rotate',): # Jangan reset rotate
            key_flags[key] = False

def key_event(key):

    global key_last
    global key_flags
    global mouse_mark
    global cal_last

    # config mode
    if key == ord('c'): # Gunakan ord() untuk kejelasan
        if key_flags['config']:
            key_flags['config'] = False
        else:
            key_flags_clear()
            key_flags['config'] = True
            cal_last,mouse_mark = 0,None # Reset kalibrasi saat masuk mode config

    # normilization mode
    elif key == ord('n'):
        if key_flags['norms']:
            key_flags['norms'] = False
        else:
            # Matikan mode lain yang mungkin aktif
            key_flags['config'] = False
            key_flags['auto'] = False
            key_flags['thresh'] = False
            key_flags['percent'] = False
            key_flags['lock'] = False
            key_flags['norms'] = True
            mouse_mark = None

    # rotate
    elif key == ord('r'):
        key_flags['rotate'] = not key_flags['rotate'] # Toggle rotate

    # auto mode
    elif key == ord('a'):
        if key_flags['auto']:
            key_flags['auto'] = False
        else:
            key_flags_clear() # Matikan mode lain
            key_flags['auto'] = True
            mouse_mark = None

    # auto percent (hanya jika auto mode aktif)
    elif key == ord('p') and key_flags['auto']:
        key_flags['percent'] = not key_flags['percent']
        key_flags['thresh'] = False # Matikan thresh jika percent aktif
        key_flags['lock'] = False

    # auto threshold (hanya jika auto mode aktif)
    elif key == ord('t') and key_flags['auto']:
        key_flags['thresh'] = not key_flags['thresh']
        key_flags['percent'] = False # Matikan percent jika thresh aktif
        key_flags['lock'] = False

    # log
    # Hindari error jika key bukan karakter printable
    try:
        key_char = chr(key)
    except ValueError:
        key_char = 'N/A'
    print(f'key: [{key}, {key_char}] flags: {key_flags}')
    key_last = key

#-------------------------------
# mouse events
#-------------------------------

# mouse events
mouse_raw  = (0,0) # pixels from top left
mouse_now  = (0,0) # pixels from center
mouse_mark = None  # last click (from center)

# mouse callback
def mouse_event(event,x,y,flags,parameters):

    #print(event,x,y,flags,parameters)

    # event =  0 = current location
    # event =  1 = left   down click
    # event =  2 = right  down click
    # event =  3 = middle down
    # event =  4 = left   up   click
    # event =  5 = right  up   click
    # event =  6 = middle up
    # event = 10 = middle scroll, flag negative|positive value = down|up

    # globals
    global mouse_raw
    global mouse_now
    global mouse_mark
    global key_last
    global auto_percent
    global auto_threshold
    global auto_blur
    global norm_alpha
    global norm_beta

    # Pastikan x, y dalam batas frame
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    # update percent (jika mode aktif)
    if key_flags['percent']:
        # Gunakan rentang yang lebih intuitif, misal 0-10%
        auto_percent = 10 * (x / width)
        print(f"Mouse Update: auto_percent = {auto_percent:.2f}")


    # update threshold (jika mode aktif)
    elif key_flags['thresh']:
        auto_threshold = int(255*x/width)
        # Blur ganjil antara 1 dan 21
        raw_blur = int(20*y/height)
        auto_blur = (raw_blur // 2) * 2 + 1 # Pastikan ganjil
        auto_blur = max(1, min(auto_blur, 21)) # Batasi antara 1 dan 21
        print(f"Mouse Update: auto_threshold = {auto_threshold}, auto_blur = {auto_blur}")


    # update normalization (jika mode aktif)
    elif key_flags['norms']:
        # Rentang alpha 0-64, beta 128-255
        norm_alpha = int(64*x/width)
        norm_beta  = max(norm_alpha + 1, min(255,int(128+(127*y/height)))) # Pastikan beta > alpha
        print(f"Mouse Update: norm_alpha = {norm_alpha}, norm_beta = {norm_beta}")


    # update mouse location (selalu)
    mouse_raw = (x,y)

    # offset from center
    # invert y to standard quadrants
    ox = x - cx
    oy = (y-cy)*-1

    # update mouse location (pixels from center)
    # Hanya update jika tidak terkunci
    if not key_flags['lock']:
        mouse_now = (ox,oy)

    # left click event
    if event == cv2.EVENT_LBUTTONDOWN: # Gunakan konstanta event OpenCV

        # Jika mode config aktif, tandai titik untuk kalibrasi
        if key_flags['config']:
            key_flags['lock'] = False # Tidak perlu lock di config mode
            mouse_mark = (ox,oy)
            print(f"Config Click: Marked at {mouse_mark} for D={cal_last}")


        # Jika mode auto aktif, tidak ada aksi khusus saat klik kiri
        # (kecuali mematikan mode setting percent/thresh)
        elif key_flags['auto']:
            if key_flags['percent'] or key_flags['thresh']:
                 key_flags['percent'] = False
                 key_flags['thresh'] = False
                 print("Auto Settings Mode Deactivated by Click")
            else:
                # Mungkin bisa digunakan untuk memilih kontur? (fitur tambahan)
                pass


        # Jika mode normalize aktif, matikan mode setting
        elif key_flags['norms']:
            key_flags['norms'] = False
            print("Normalize Settings Mode Deactivated by Click")


        # Jika mode dimensi standar
        else:
            if not key_flags['lock']:
                # Jika sudah ada tanda, kunci pengukuran
                if mouse_mark:
                    key_flags['lock'] = True
                    print("Dimension Locked")
                # Jika belum ada tanda, buat tanda pertama
                else:
                    mouse_mark = (ox,oy)
                    print(f"Dimension Start: Marked at {mouse_mark}")
            # Jika sudah terkunci, buka kunci dan mulai dari titik baru
            else:
                key_flags['lock'] = False
                mouse_now = (ox,oy) # Update posisi saat ini
                mouse_mark = (ox,oy) # Mulai tanda baru di lokasi klik
                print(f"Dimension Unlocked & Reset: New mark at {mouse_mark}")


        key_last = 0 # Reset key last setelah aksi mouse

    # right click event -> reset mode / pengukuran
    elif event == cv2.EVENT_RBUTTONDOWN:
        if key_flags['config'] or key_flags['auto'] or key_flags['norms']:
            key_flags_clear() # Keluar dari mode spesial
            print("Mode Reset by Right Click")
        else:
            # Reset pengukuran di mode dimensi
            mouse_mark = None
            key_flags['lock'] = False
            print("Dimension Reset by Right Click")
        key_last = 0

# register mouse callback
cv2.setMouseCallback(framename,mouse_event)

#-------------------------------
# main loop
#-------------------------------

print("\nStarting Main Loop...")
print("Keys: Q=Quit, R=Rotate, N=Normalize, A=AutoMode, C=ConfigMode")
print("In AutoMode: P=Set Min Percent, T=Set Threshold/Blur")
print("Mouse: LeftClick=Mark/Lock/Confirm, RightClick=Reset/Cancel\n")


# loop
while 1:

    # get frame
    frame0 = camera.next(wait=1)
    if frame0 is None:
        # Jika sumbernya file video dan sudah habis, keluar
        if isinstance(camera_id, str):
            print("End of video file reached.")
            break
        time.sleep(0.1)
        continue

    # --- FIX: Pastikan frame tidak kosong ---
    if frame0.size == 0:
        print("WARN: Received empty frame.")
        continue

    # normalize (jika nilai alpha/beta tidak default)
    if norm_alpha != 0 or norm_beta != 255:
        # Cek tipe data frame sebelum normalize
        if frame0.dtype != np.uint8:
            frame0 = frame0.astype(np.uint8) # Konversi jika perlu
        try:
            cv2.normalize(frame0,frame0,norm_alpha,norm_beta,cv2.NORM_MINMAX)
        except cv2.error as e:
            print(f"Normalize Error: {e}. Resetting alpha/beta.")
            norm_alpha, norm_beta = 0, 255 # Reset jika error


    # rotate 180
    if key_flags['rotate']:
        frame0 = cv2.rotate(frame0,cv2.ROTATE_180)

    # start top-left text block
    text = []

    # camera text
    fps = camera.current_frame_rate
    text.append(f'CAMERA: {camera_id} {width}x{height} {fps:.2f}FPS')

    # mouse text
    text.append('')
    if not mouse_mark:
        text.append(f'LAST CLICK: NONE')
    else:
        # Tampilkan koordinat piksel relatif dari tengah
        text.append(f'LAST CLICK: {mouse_mark} PIXELS')
    # Tampilkan koordinat piksel relatif dari tengah
    text.append(f'CURRENT XY: {mouse_now} PIXELS')

    #-------------------------------
    # normalize mode
    #-------------------------------
    if key_flags['norms']:

        # print
        text.append('')
        text.append(f'NORMALIZE MODE (Adjust with Mouse)')
        text.append(f'ALPHA (min): {norm_alpha}')
        text.append(f'BETA (max): {norm_beta}')
        text.append(f'Left Click to Confirm')

    #-------------------------------
    # config mode
    #-------------------------------
    elif key_flags['config']:

        # quadrant crosshairs
        draw.crosshairs(frame0,5,weight=2,color='red',invert=True)

        # crosshairs aligned (rotated) to maximum distance
        # Gambar garis diagonal utama saja untuk kejelasan
        draw.line(frame0,cx,cy, width, height,weight=1,color='red') # Bawah Kanan
        draw.line(frame0,cx,cy, 0, height,weight=1,color='red')     # Bawah Kiri
        draw.line(frame0,cx,cy, width, 0,weight=1,color='red')      # Atas Kanan
        draw.line(frame0,cx,cy, 0, 0,weight=1,color='red')          # Atas Kiri


        # mouse cursor lines (simple crosshair at mouse)
        mx,my = mouse_raw
        draw.hline(frame0, my, weight=1, color='green')
        draw.vline(frame0, mx, weight=1, color='green')

        # config text data
        text.append('')
        text.append(f'CONFIG MODE')
        text.append(f'Target Range: {cal_range}{unit_suffix}')

        caltext = "" # Inisialisasi caltext

        # start cal
        if cal_last == 0: # Baru masuk mode config
            cal_last = cal_base
            caltext = f'CONFIG: Click on D = {cal_last}{unit_suffix}'

        # continue cal
        elif cal_last <= cal_range:
            # Jika ada klik mouse, proses update kalibrasi
            if mouse_mark:
                # Pastikan mouse_mark tidak None sebelum unpack
                mx_cal, my_cal = mouse_mark
                cal_update(mx_cal, my_cal, cal_last)
                cal_last += cal_base
                mouse_mark = None # Reset mouse_mark setelah diproses

            # Tampilkan instruksi berikutnya atau final
            current_target_d = min(cal_last, cal_range) # Tampilkan target saat ini atau maks
            caltext = f'CONFIG: Click on D = {current_target_d}{unit_suffix}'
            if current_target_d == cal_range and cal_last > cal_range:
                 caltext += ' (Final Point)'


        # done
        else:
            # Simpan kalibrasi setelah selesai
            print("CONFIG: Calibration complete. Saving...")
            try:
                with open(calfile,'w') as f:
                    data = list(cal.items())
                    data.sort() # Urutkan berdasarkan piksel
                    for key,value in data:
                        f.write(f'd,{key},{value}\n')
                    print(f"Calibration saved to {calfile}")
            except Exception as e:
                print(f"ERROR: Could not save calibration file {calfile}. Error: {e}")

            caltext = f'CONFIG: Complete. Saved.'
            key_flags['config'] = False # Keluar dari mode config otomatis
            cal_last = None # Reset status kalibrasi


        # Tambahkan caltext ke blok teks utama
        if caltext: # Hanya tambahkan jika ada isinya
             text.append(caltext)

        # Hapus mouse mark setelah diproses atau jika tidak relevan lagi
        # mouse_mark = None # Sudah direset di dalam blok 'continue cal'

    #-------------------------------
    # auto mode
    #-------------------------------
    elif key_flags['auto']:

        # Tampilkan status setting jika aktif
        if key_flags['percent']:
             text.append('')
             text.append(f'AUTO SETTING: MIN PERCENT (Adjust X)')
             text.append(f'Current: {auto_percent:.2f}%')
             text.append(f'Left Click to Confirm')
        elif key_flags['thresh']:
             text.append('')
             text.append(f'AUTO SETTING: THRESH/BLUR (Adjust X/Y)')
             text.append(f'Threshold: {auto_threshold}')
             text.append(f'Gauss Blur: {auto_blur}')
             text.append(f'Left Click to Confirm')
        else:
            # Tampilkan parameter auto mode standar
            text.append('')
            text.append(f'AUTO MODE')
            text.append(f'UNITS: {unit_suffix}')
            text.append(f'MIN PERCENT: {auto_percent:.2f}')
            text.append(f'THRESHOLD: {auto_threshold}')
            text.append(f'GAUSS BLUR: {auto_blur}')

        # gray frame
        frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

        # blur frame
        # Pastikan blur kernel size valid (ganjil > 0)
        blur_ksize = (auto_blur, auto_blur)
        if blur_ksize[0] > 0 and blur_ksize[1] > 0:
             frame1 = cv2.GaussianBlur(frame1,blur_ksize,0)
        else:
             print("WARN: Invalid blur size, skipping GaussianBlur.")


        # threshold frame n out of 255
        # Pastikan threshold value valid (0-255)
        thresh_val = max(0, min(auto_threshold, 255))
        frame1 = cv2.threshold(frame1,thresh_val,255,cv2.THRESH_BINARY)[1]

        # invert (agar objek putih, background hitam)
        frame1 = ~frame1

        # find contours on thresholded image
        contours,nada = cv2.findContours(frame1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # small crosshairs (setelah mendapatkan frame1)
        draw.crosshairs(frame0,5,weight=2,color='green')

        # loop over the contours
        for c in contours:

            # contour data (from top left)
            x1,y1,w,h = cv2.boundingRect(c)
            # Abaikan kontur yang terlalu kecil di level piksel
            if w < 5 or h < 5: continue

            x2,y2 = x1+w,y1+h
            x3,y3 = x1+(w/2),y1+(h/2) # Titik tengah bounding box (piksel)

            # percent area
            # --- FIX: Gunakan variabel area ---
            percent = 100*w*h/area

            # if the contour is too small, ignore it
            if percent < auto_percent:
                    continue

            # if the contour is too large, ignore it
            elif percent > 95: # Mungkin 60 terlalu kecil, naikkan batas atas
                    continue

            # convert to center, then distance
            # Konversi pojok-pojok bounding box
            # Perlu diingat konversi ini mengukur jarak dari pusat frame ke pojok tsb
            # Untuk mendapatkan panjang/lebar objek, lebih baik konversi titik tengah
            # dan gunakan w, h dalam piksel lalu konversi. Atau konversi x1,y1 dan x2,y2
            # lalu hitung selisihnya.

            # Metode 1: Konversi pojok-pojok
            # x1_conv, y1_conv = conv(x1 - cx, (y1 - cy)*-1)
            # x2_conv, y2_conv = conv(x2 - cx, (y2 - cy)*-1)
            # xlen = abs(x1_conv - x2_conv) # Ini tidak selalu benar karena proyeksi
            # ylen = abs(y1_conv - y2_conv)

            # Metode 2: Konversi lebar dan tinggi piksel (lebih disarankan)
            # Asumsi skala relatif konstan di area kontur kecil
            # Ambil skala di titik tengah kontur
            mid_x_rel, mid_y_rel = x3 - cx, (y3 - cy)*-1
            dist_mid = hypot(mid_x_rel, mid_y_rel)
            scale_mid = cal[baseround(dist_mid, pixel_base)] if baseround(dist_mid, pixel_base) in cal else (cal_range / dm)

            xlen = w * scale_mid
            ylen = h * scale_mid


            alen = 0 # Rata-rata sisi (jika mendekati persegi)
            if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                alen = (xlen+ylen)/2
            # --- FIX: Gunakan nama variabel carea ---
            carea = xlen*ylen

            # --- AWAL: KLASIFIKASI BERDASARKAN AREA (AUTO MODE) ---
            kapasitas_prediksi = "Tidak Diketahui" # Default value

            # !!! PENTING: Sesuaikan rentang nilai area (mm^2) di bawah ini
            # berdasarkan hasil pengujian Anda dengan botol sebenarnya!
            # Contoh rentang (HARUS DIGANTI):
            if 196 < carea < 219:  # Rentang contoh untuk botol ~330ml?
                kapasitas_prediksi = "450ml (Orson)"
            elif 219 < carea < 250: # Rentang contoh untuk botol 500ml/600ml?
                kapasitas_prediksi = "425ml (Marjan)"
            elif 187 < carea < 196: # Rentang contoh untuk botol 1000ml/1500ml?
                kapasitas_prediksi = "500ml (Prediksi)"
            # Tambahkan 'elif' lain jika perlu untuk kapasitas berbeda

            # --- AKHIR: KLASIFIKASI BERDASARKAN AREA ---


            # plot bounding box
            draw.rect(frame0,x1,y1,x2,y2,weight=2,color='red')

            # add dimensions
            # Tampilkan X Len di atas
            draw.add_text(frame0,f'{xlen:.2f}',x3,y1-8,center=True,color='red')
            # --- FIX: Tambahkan teks "Area:" ---
            draw.add_text(frame0,f'Area: {carea:.2f}',x3,y2+8,center=True,top=True,color='red')

            # Tambahkan teks prediksi di bawah teks Area
            draw.add_text(frame0, f'Prediksi: {kapasitas_prediksi}', x3, y2 + 24, center=True, top=True, color='blue')

            # Tampilkan Y Len di samping (pilih sisi yang lebih kosong)
            y_mid = y1 + h/2
            if x1 < width-x2: # Lebih banyak ruang di kanan
                draw.add_text(frame0,f'{ylen:.2f}',x2+4,y_mid,middle=True,color='red')
            else: # Lebih banyak ruang di kiri
                draw.add_text(frame0,f'{ylen:.2f}',x1-4,y_mid,middle=True,right=True,color='red')

            # Tampilkan Avg jika ada
            if alen:
                 # Sesuaikan posisi Y jika perlu
                draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y2+40,center=True,top=True,color='green')


    #-------------------------------
    # dimension mode
    #-------------------------------
    else: # Jika tidak dalam mode config, auto, atau norms

        # small crosshairs
        draw.crosshairs(frame0,5,weight=2,color='green')

        # mouse cursor lines
        draw.vline(frame0,mouse_raw[0],weight=1,color='green')
        draw.hline(frame0,mouse_raw[1],weight=1,color='green')

        # draw measurement if a mark exists
        if mouse_mark:

            # locations (relative to center)
            x1_rel,y1_rel = mouse_mark
            x2_rel,y2_rel = mouse_now

            # convert relative pixel coordinates to real-world units
            x1c,y1c = conv(x1_rel,y1_rel)
            x2c,y2c = conv(x2_rel,y2_rel)

            # calculate lengths in real-world units
            xlen = abs(x1c-x2c)
            ylen = abs(y1c-y2c)
            llen = hypot(xlen,ylen) # Diagonal length

            alen = 0 # Average side length (if close to square)
            if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                alen = (xlen+ylen)/2
            # --- FIX: Gunakan nama variabel carea ---
            carea = xlen*ylen # Area of the bounding box in units^2

            # --- AWAL: KLASIFIKASI BERDASARKAN AREA (DIMENSION MODE) ---
            kapasitas_prediksi = "Tidak Diketahui" # Default value

            # !!! PENTING: Sesuaikan rentang nilai area (mm^2) di bawah ini
            # Gunakan rentang yang SAMA dengan yang di Auto Mode.
            # Contoh rentang (HARUS DIGANTI):
            if 5000 < carea < 9000:  # Rentang contoh untuk botol ~330ml?
                kapasitas_prediksi = "330ml (Prediksi)"
            elif 10000 < carea < 16000: # Rentang contoh untuk botol 500ml/600ml?
                kapasitas_prediksi = "500/600ml (Prediksi)"
            elif 18000 < carea < 28000: # Rentang contoh untuk botol 1000ml/1500ml?
                kapasitas_prediksi = "1000/1500ml (Prediksi)"
            # Tambahkan 'elif' lain jika perlu untuk kapasitas berbeda

            # --- AKHIR: KLASIFIKASI BERDASARKAN AREA ---


            # print distances to the main text block
            text.append('')
            text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
            text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
            text.append(f'L LEN: {llen:.2f}{unit_suffix}')
            # --- FIX: Tambahkan Area dan Prediksi ke teks utama ---
            text.append(f'AREA: {carea:.2f}{unit_suffix}^2')
            text.append(f'PREDIKSI: {kapasitas_prediksi}')


            # convert relative coordinates back to absolute pixel coordinates for drawing
            x1_abs = x1_rel + cx
            y1_abs = (y1_rel * -1) + cy
            x2_abs = x2_rel + cx
            y2_abs = (y2_rel * -1) + cy

            # Calculate midpoint and bottom y for text placement
            x_mid_abs = x1_abs+((x2_abs-x1_abs)/2)
            y_bottom_abs = max(y1_abs,y2_abs)
            y_top_abs = min(y1_abs, y2_abs)
            y_mid_abs = y1_abs + ((y2_abs - y1_abs)/2)


            # line weight
            weight = 1
            if key_flags['lock']:
                weight = 2

            # plot rectangle and diagonal line
            draw.rect(frame0,x1_abs,y1_abs,x2_abs,y2_abs,weight=weight,color='red')
            draw.line(frame0,x1_abs,y1_abs,x2_abs,y2_abs,weight=weight,color='green')

            # add dimensions text onto the frame
            # X Length above the box
            draw.add_text(frame0,f'{xlen:.2f}',x_mid_abs, y_top_abs - 8, center=True, color='red')
            # --- FIX: Tambahkan teks "Area:" ---
            draw.add_text(frame0,f'Area: {carea:.2f}',x_mid_abs, y_bottom_abs + 8, center=True, top=True, color='red')

            # Y Length beside the box (choose side with more space)
            if x2_abs <= x1_abs: # Box drawn right to left
                draw.add_text(frame0,f'{ylen:.2f}',x1_abs+4, y_mid_abs, middle=True, color='red')
                # Diagonal length near the end point (bottom-left)
                draw.add_text(frame0,f'{llen:.2f}',x2_abs-4, y2_abs+15, right=True, color='green')
            else: # Box drawn left to right
                draw.add_text(frame0,f'{ylen:.2f}',x1_abs-4, y_mid_abs, middle=True, right=True, color='red')
                 # Diagonal length near the end point (bottom-right)
                draw.add_text(frame0,f'{llen:.2f}',x2_abs+8, y2_abs+15, color='green')

            # Average length if applicable
            if alen:
                draw.add_text(frame0,f'Avg: {alen:.2f}',x_mid_abs, y_bottom_abs + 24, center=True, top=True, color='green')


    # add usage key help text
    text.append('')
    text.append(f'Q = QUIT')
    text.append(f'R = ROTATE')
    text.append(f'N = NORMALIZE')
    text.append(f'A = AUTO-MODE')
    if key_flags['auto']:
        text.append(f' P = MIN-PERCENT (Adj X)')
        text.append(f' T = THRESH/BLUR (Adj X/Y)')
    text.append(f'C = CONFIG-MODE')
    if key_flags['lock']:
         text.append(f'LOCKED (Left Click Unlock)')
    elif mouse_mark and not key_flags['auto'] and not key_flags['config'] and not key_flags['norms']:
         text.append(f'MARK SET (Left Click Lock)')


    # draw top-left text block
    draw.add_text_top_left(frame0,text)

    # display the frame
    cv2.imshow(framename,frame0)

    # key delay and action
    key = cv2.waitKey(1) & 0xFF

    # esc ==  27 == quit
    # q   == 113 == quit
    if key in (27,ord('q')):
        print("Quit key pressed. Exiting...")
        break

    # process other keys if pressed
    # elif key != 255: # 255 seringkali berarti tidak ada tombol ditekan di Linux
    elif key != 255 and key != -1 : # -1 seringkali berarti tidak ada tombol ditekan di Windows/Mac
        key_event(key)

#-------------------------------
# kill sequence
#-------------------------------

print("Exiting application...")

# close camera thread
if 'camera' in locals() and camera.frame_grab_on:
    print("Stopping camera thread...")
    camera.stop()

# close all windows
print("Closing OpenCV windows...")
cv2.destroyAllWindows()

# done
print("Done.")
exit()

#-------------------------------
# end
#-------------------------------

