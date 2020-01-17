import cv2
from networktables import NetworkTables
from networktables import NetworkTableInstance

class Processing:
	def __init__(self):
		#blur#
		self.blur_radius = 5
		
		#resize image#
		self.width = 320.0
		self.height = 240.0
		self.interpolation = cv2.INTER_CUBIC
		self.output = None
		self.input = None
		
		#hsv threshold#
		self.hue = [0.0, 180.0]
		self.saturation = [0.0, 32.0]
		self.value = [217.0, 255.0]
		
		#find contour#
		self.min_area = 9.0
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
		
	def blur_image(frame, radius):
		return cv2.blur(frame,(5,5))
		
	def resize_image(frame, width, height, interpolation):
		return cv2.resize(frame, ((int)(width), (int)(height)), 0, 0, interpolation)
		
	def hsv_threshold(frame, hue, sat, val):
		output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		return cv2.inRange(output, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))
		
	def find_contours(input, external_only):
		if(external_only):
			mode = cv2.RETER_EXTERNAL
		else:
			mode = cv2.RETR_LIST
		method = cv2.CHAIN_APPROX_SIMPLE
		_, contours, _ = cv2.findContours(input, mode=mode, method=method)
		return contours
		
	def filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,min_height, max_height, solidity, max_vertex_count, min_vertex_count,min_ratio, max_ratio):
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
		self.output = self.hsv_threshold(self.input, self.hue, self.saturation, self.value)
		
		self.input = self.output
		self.output = self.find_contours(self.input, False)
		
		self.input = self.output
		self.output = self.filter_contours(self.output, self.min_area, self.min_perimeter, self.min_width, self.max_width, self.min_height, self.max_height, self.solidity, self.max_vertices, self.min_vertices, self.min_ratio, self.max_ratio)
		
def main():
	init = NetworkTableInstance.getDefault()
	init.starClientTeam(8180)
	table = NetworkTables.getTable('/datatable')
	
	pipline = Processing()
	
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		time, frame = cap.read()
		if time:
			pipeline.process(frame)
			
			for contour in pipeline.output:
				x, y, w, h = cv2.boundingRect(contour)
				center_x = x + w/2
				center_y = y + h/2
				widths = w
				heights = h
				table.putNumber('x', center_x)
				table.putNumber('y', center_y)
				table.putNumber('width', widths)
				table.putNumber('height', heights)	

if __name__ == '__main__':
	main()