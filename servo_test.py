from adafruit_servokit import ServoKit
import board
import busio
import time

i2c_bus0 = (busio.I2C(board.SCL_1,board.SDA_1))
kit = ServoKit(channels=16,i2c=i2c_bus0)

while True:
    angle = int(input("set servo angle to: "))
    if 0<=angle<=180:
        kit.servo[0].angle = angle
    else:
        print("angle out of range! try again")
