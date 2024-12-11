import cv2
import sys
from image import MyImage

def main():
	if len(sys.argv) < 2:
		print("Usage: python3 main.py <filename>")
		exit(1)

	filename = sys.argv[1]
	image = MyImage(filename)

	while True:
		print("\nAvailable commands:")
		print("ADJUST commands: CROP, ROTATE, FLIP, RESIZE, BRIGHTNESS, CONTRAST")
		print("FILTER commands: APPLY, GRAYSCALE, EQUALIZE, UNBLUR")
		print("ADVANCED: BLEND, PALM, SIFT, RANSAC, DETECT_FACES, RECOGNIZE_FACES")
		print("UTILS commands: SHOW, SAVE, EXIT")

		command = input("Enter command: ").strip().upper()

		if command == "CROP":
			x1, y1, x2, y2 = map(int, input("Enter coordinates (x1, y1, x2, y2): ").split())
			image.crop(x1, y1, x2, y2)
		
		elif command == "ROTATE":
			angle = int(input("Enter angle to rotate: "))
			image.rotate(angle)

		elif command == "FLIP":
			direction = int(input("Enter flip direction (0=vertical, 1=horizontal, -1=both): "))
			image.flip(direction)

		elif command == "RESIZE":
			image.resize()
		
		elif command == "BRIGHTNESS":
			brightness = int(input("Enter brightness value (-100 to 100): "))
			image.adjust_brightnes_contrast(brightness=brightness)

		elif command == "CONTRAST":
			contrast = float(input("Enter contrast value (0.5 to 3.0): "))
			image.adjust_brightnes_contrast(contrast=contrast)
		
		# FILTERS

		elif command == "APPLY":
			filter_type = input("Enter filter (BLUR, MEDIAN_BLUR, SHARPEN, EDGE): ").strip().upper()
			if filter_type == "BLUR" or filter_type == "MEDIAN_BLUR":
				kernel_size = int(input("Enter kernel size: "))
				sigmaX = float(input("Enter standard deviation for X: "))
				image.apply_filter(filter_type, kernel_size=kernel_size, sigmaX=sigmaX)
			elif filter_type == "SHARPEN":
				intensity = float(input("Enter intnsity of the sharpening (0-3): "))
				denoise_strength = float(input("Enter denoise strength (0-30): "))
				image.apply_filter(filter_type, intensity=intensity, denoise_strength=denoise_strength)
			else:
				image.apply_filter(filter_type)

		elif command == "GRAYSCALE":
			image.gray_scale()

		elif command == "EQUALIZE":
			image.equalize()
		
		elif command == "UNBLUR":
			image.unblur()

		# Machine-Learning algorithms

		elif command == "BLEND":
			image.blend()

		elif command == "SIFT":
			x = input("Enter First Image Name: ").strip()
			image1 = cv2.imread(x)
			if image1 is None:
				raise FileNotFoundError(f"Image '{x}' not found")

			y = input("Enter Second Image Name: ").strip()
			image2 = cv2.imread(y)
			if image2 is None:
				raise FileNotFoundError(f"Image '{y}' not found")

			out_filename = input("Enter output filename: ").strip()
			image.sift(filename1=x, filename2=y, out_filename=out_filename)

		elif command == "RANSAC":
			img_source = cv2.imread("test1.png")
			img_target = cv2.imread("test2.png")
			output_file = input("Enter output filename: ").strip()
			image.ransac(img1=img_source, img2=img_target, output_file=output_file)

		elif command == "PALM":
			image.find_palm_lines()
		
		elif command == "DETECT_FACES":
			image.detect_face()

		elif command == "RECOGNIZE_FACES":
			image.recognize_faces()

		# UTILS:
		elif command == "SHOW":
			image.show()

		elif command == "SAVE":
			outputFilename = input("Enter filename to save: ")
			image.save(outputFilename)
		
		elif command == "EXIT":
			print("Exiting the application.")
			break

		else:
			print("Invalid command. Please try again.")

if __name__ == "__main__":
	main()
