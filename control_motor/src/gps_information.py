import serial

# 시리얼 포트 설정 (포트와 보드레이트를 수정할 수 있습니다)
port = '/dev/ttyS0'  # 또는 '/dev/ttyAMA0'
ser = serial.Serial(port, 9600, timeout=1)
ser.flush()

def parse_nmea(sentence):
    parts = sentence.split(',')
    if parts[0] == '$GPGGA':
        utc_time = parts[1] if parts[1] else "정보 없음"
        latitude = f"{parts[2]} {parts[3]}" if parts[2] and parts[3] else "정보 없음"
        longitude = f"{parts[4]} {parts[5]}" if parts[4] and parts[5] else "정보 없음"
        fix_quality = parts[6] if parts[6] else "정보 없음"
        num_sats = parts[7] if parts[7] else "정보 없음"
        hdop = parts[8] if parts[8] else "정보 없음"
        altitude = f"{parts[9]} {parts[10]}" if parts[9] and parts[10] else "정보 없음"

        print(f"UTC 시간: {utc_time}")
        print(f"위도: {latitude}")
        print(f"경도: {longitude}")
        print(f"위치 품질: {fix_quality}")
        print(f"위성 수: {num_sats}")
        print(f"수평 정밀도(Horizontal Dilution of Precision, HDOP): {hdop}")
        print(f"고도: {altitude}")

    elif parts[0] == '$GPRMC':
        utc_time = parts[1] if parts[1] else "정보 없음"
        latitude = f"{parts[3]} {parts[4]}" if parts[3] and parts[4] else "정보 없음"
        longitude = f"{parts[5]} {parts[6]}" if parts[5] and parts[6] else "정보 없음"
        speed_knots = parts[7] if parts[7] else "정보 없음"

        print(f"UTC 시간: {utc_time}")
        print(f"위도: {latitude}")
        print(f"경도: {longitude}")
        print(f"지면 속도(노트 단위): {speed_knots}")

    elif parts[0] == '$GPVTG':
        course_over_ground = parts[1] if parts[1] else "정보 없음"
        speed_knots = parts[7] if parts[7] else "정보 없음"

        print(f"진북 기준 방향: {course_over_ground}")
        print(f"지면 속도(노트 단위): {speed_knots}")

    else:
        print(f"수신된 문장: {sentence}")
    print("")

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        if line.startswith('$'):
            parse_nmea(line)
