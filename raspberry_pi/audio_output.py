import serial
import time

class DFPlayer:
    def __init__(self, port="/dev/serial0", baudrate=9600):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=1)
        time.sleep(2)
        self.set_volume(30)

    def send_command(self, cmd, param1=0, param2=0):
        command_line = bytearray([
            0x7E,
            0xFF,
            0x06,
            cmd,
            0x00,
            param1,
            param2,
            0x00,
            0x00,
            0xEF
        ])
        checksum = 0 - sum(command_line[1:7])
        checksum &= 0xFFFF
        command_line[7] = (checksum >> 8) & 0xFF
        command_line[8] = checksum & 0xFF
        self.ser.write(command_line)
        time.sleep(0.2)

    def set_volume(self, volume):
        volume = max(0, min(volume, 30))
        self.send_command(0x06, 0x00, volume)

    def play_track(self, folder=1, file_number=1):
        if 1 <= file_number <= 3000:
            self.send_command(0x12, 0x00, file_number)

    def stop(self):
        self.send_command(0x16)

    def pause(self):
        self.send_command(0x0E)

    def resume(self):
        self.send_command(0x0D)

    def __del__(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

