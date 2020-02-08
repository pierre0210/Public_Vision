#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen/releases
#-----------------------------------------------------------------------------
import cv2
from networktables import NetworkTables
from networktables import NetworkTablesInstance
from cscore import CameraServer, VideoSource
import numpy as np
import json
import sys
from threading import Thread
import math
import time

blur_radius = 2

#image size ratioed to 4:3
image_width = 160
image_height = 120

H_Aspect = 4
V_Aspect = 3

#target
real_width = 96
real_height = 39

#Lifecam3000
diagonalView = math.radians(68.5)

diagonalAspect = math.hypot(H_Aspect, V_Aspect)

#FOV(Field of view)
horizontalView = math.atan(math.tan(diagonalView/2) * (H_Aspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (V_Aspect / diagonalAspect)) * 2

camera_center_X = image_width/2 - .5
camera_center_Y = image_height/2 - .5

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))

lower_color = np.array([25.0, 180.0, 30.0])
upper_color = np.array([180.0, 255.0, 255.0])

class Processing:
	def __init__(self, image_width, image_height, lower_color, upper_color, blur):
		#blur#
		self.blur_radius = blur
		
		#resize image#
		self.width = image_width
		self.height = image_height
		self.interpolation = cv2.INTER_CUBIC
		self.output = None
		self.input = None
		
		#hsv threshold#
		self.lower_color = lower_color
		self.upper_color = upper_color
		
		#find contour#
		self.min_area = 80.0
		self.min_perimeter = 30.0
		self.min_width = 10.0
		self.max_width = 1000
		self.min_height = 20.0
		self.max_height = 1000
		self.solidity = [0.0, 100.0]
		self.max_vertices = 1000000
		self.min_vertices = 0
		self.min_ratio = 0
		self.max_ratio = 1000
		
	def blur_image(self, frame, radius):
		return cv2.blur(frame,(radius,radius))
		
	def resize_image(self, frame, width, height, interpolation):
		return cv2.resize(frame, ((int)(width), (int)(height)), 0, 0, interpolation)
		
	def hsv_threshold(self, frame, lower_color, upper_color):
		output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		return cv2.inRange(output, lower_color, upper_color)
		
	def find_contours(self, input, external_only):
		if(external_only):
			mode = cv2.RETR_EXTERNAL
		else:
			mode = cv2.RETR_LIST
		method = cv2.CHAIN_APPROX_SIMPLE
		_, contours, _ = cv2.findContours(input, mode=mode, method=method) #warning!!!! when testing in rpi, you should add one more " _, " at the front
		return contours
		
	def filter_contours(self, input_contours, min_area, min_perimeter, min_width, max_width,min_height, max_height, solidity, max_vertex_count, min_vertex_count,min_ratio, max_ratio):
		output = []
		for contour in input_contours:
			x,y,w,h = cv2.boundingRect(contour)
			if (w < min_width or w > max_width):
				continue
			if (h < min_height or h > max_height):
				continue
			area = cv2.contourArea(contour)
			if (area < min_area):
				continue
			if (cv2.arcLength(contour, True) < min_perimeter):
				continue
			hull = cv2.convexHull(contour)
			solid = 100 * area / cv2.contourArea(hull)
			if (solid < solidity[0] or solid > solidity[1]):
				continue
			if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
				continue
			ratio = (float)(w) / h
			if (ratio < min_ratio or ratio > max_ratio):
				continue
			output.append(contour)
		return output
		
	def process(self, source0):
		self.input = source0
		self.output = self.blur_image(self.input, self.blur_radius)
		
		self.input = self.output
		self.output = self.resize_image(self.input, self.width, self.height, self.interpolation)
		
		self.input = self.output
		self.output = self.hsv_threshold(self.input, self.lower_color, self.upper_color)
		
		self.input = self.output
		self.output = self.find_contours(self.input, True)
		
		self.input = self.output
		self.output = self.filter_contours(self.output, self.min_area, self.min_perimeter, self.min_width, self.max_width, self.min_height, self.max_height, self.solidity, self.max_vertices, self.min_vertices, self.min_ratio, self.max_ratio)

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None, name='stream'):
        self.outputStream = cameraServer.putVideo(name, imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True

    def notifyError(self, error):
        self.outputStream.notifyError(error)

class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        #Automatically sets exposure to 0 to track tape
        self.webcam = camera
        self.webcam.setExposureManual(0)
        #Some booleans so that we don't keep setting exposure over and over to the same value
        self.autoExpose = False
        self.prevValue = self.autoExpose
        #Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        #Gets the video
        self.stream = cameraServer.getVideo(camera = camera)
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            #Boolean logic we don't keep setting exposure over and over to the same value
            if self.autoExpose:

                self.webcam.setExposureAuto()
            else:

                self.webcam.setExposureManual(0)
            #gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def getError(self):
        return self.stream.getError()	

configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

def main():
	if len(sys.argv) >= 2:
		configFile = sys.argv[1]
	if not readConfig():
		sys.exit(1)
	
	init = NetworkTablesInstance.getDefault()
	
	if server:
		print("Setting up server...")
		init.startServer()
	else:
		print("Setting up NetworkTables client for team {}".format(team))
		init.startClientTeam(team)
	
	table = NetworkTables.getTable('PublicVision')
	
	cameras = []
	streams = []
	
	for cameraConfig in cameraConfigs:
		cs, cameraCapture = startCamera(cameraConfig)
		streams.append(cs)
		cameras.append(cameraCapture)
	
	webcam = cameras[0]
	cameraServer = streams[0]
	print("Processing...")
	table.putBoolean('connect', True)
	
	pipeline = Processing(image_width, image_height, lower_color, upper_color, blur_radius)
	
	cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()
	img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
	streamViewer = VideoShow(image_width,image_height, cameraServer, frame=img, name="PublicVision").start()
	
	while True:
		time_, frame = cap.read()
		#start = time.time()
		if time_:
			pipeline.process(frame)
			contours = sorted(pipeline.output, key=lambda x: cv2.contourArea(x), reverse=True)
			for contour in contours:
				x, y, w, h = cv2.boundingRect(contour)
				center_x = x + w/2
				center_y = y
				widths = w
				heights = h
				
				distance = (H_FOCAL_LENGTH*real_width)/widths
				H_ANGLE_TO_TARGET = math.degrees(math.atan((center_x-camera_center_X)/H_FOCAL_LENGTH))
				V_ANGLE_TO_TARGET = math.degrees(math.atan((center_y-camera_center_Y)/V_FOCAL_LENGTH))
				area = cv2.contourArea(contour)
				#print("Center_x: ",center_x)
				#print("Center_y: ",center_y)
				#print("Width: ",widths)
				#print("Heights: ",heights)
				#print("Distance", distance)
				print("H_Angle", H_ANGLE_TO_TARGET)
				#print("V_Angle", V_ANGLE_TO_TARGET)
				#now = time.time()
				#print("time: ", time_)
				#print("area", area)
				
				table.putNumber('h_angle', H_ANGLE_TO_TARGET)
				table.putNumber('v_angle', V_ANGLE_TO_TARGET)
				table.putNumber('distance', distance)
				table.putNumber('x', center_x)
				table.putNumber('y', center_y)
				table.putNumber('width', widths)
				table.putNumber('height', heights)
				#table.putNumber('time', time_)
				

if __name__ == '__main__':
	main()