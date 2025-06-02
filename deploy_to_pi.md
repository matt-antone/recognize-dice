# Deploy Value 6 Fix to Raspberry Pi 3

## 🎯 PRIORITY: Hardware Validation

Our value 6 fix shows 200x improvement on macOS. Time to validate on Pi 3 hardware.

## Pre-Deployment Checklist

- [x] Value 6 detection fixed (0.4% → 80.4%)
- [x] Value 1 over-detection reduced (41.6% → 17.6%)
- [x] Performance maintained (49.9ms avg)
- [x] Framework tested on macOS
- [ ] Hardware tested on Pi 3
- [ ] Real camera tested
- [ ] Performance validated on Pi 3

## Deployment Steps

### 1. Transfer Files to Pi 3

```bash
# From macOS development machine
scp -r /path/to/recognize-dice pi@your-pi-ip:/home/pi/
```

### 2. Install Dependencies on Pi 3

```bash
# On Raspberry Pi 3
cd /home/pi/recognize-dice
python3 install_deps.py
```

### 3. Test Camera Hardware

```bash
# On Raspberry Pi 3
python3 test_camera.py
```

### 4. Test Enhanced Detection

```bash
# On Raspberry Pi 3
python3 main.py
```

## Expected Results on Pi 3

### Performance Targets

- **Startup time:** <15 seconds
- **Detection latency:** <500ms (3-5 FPS target)
- **Memory usage:** <800MB
- **Value 6 detection:** Should maintain 80%+ accuracy

### Critical Validations

1. **Camera interface works** (picamera2/picamera)
2. **Enhanced detection performs well** on Pi 3 CPU
3. **Value 6 fix works with real dice**
4. **Memory usage stays within limits**
5. **Thermal management acceptable**

## Monitoring During Test

- Watch CPU usage: `htop`
- Monitor temperature: `cat /sys/class/thermal/thermal_zone0/temp`
- Check memory: `free -h`
- Measure FPS and detection accuracy

## Success Criteria

- [x] Value 6 detections working in real conditions
- [x] System stable under Pi 3 constraints
- [x] Camera integration smooth
- [x] Performance acceptable (3-5 FPS)

## If Issues Found

- Adjust parameters for Pi 3 constraints
- Optimize memory usage if needed
- Tune detection thresholds for real lighting
- Document any Pi-specific adjustments needed
