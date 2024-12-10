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
		print("UTILS commands: SHOW, SAVE, EXIT")
		print("ADVANCED: BLEND, SIFT, PALM, DETECT_FACES, ...")

		# TODO add the other commands 
		command = input("Enter command: ").strip().upper()

		# TODO ADD SEPARATING MENU FOR ADJUSTING, FILTERS AND THE OTHERS
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
			image.apply_filter(filter_type)
		
		elif command == "GRAYSCALE":
			image.gray_scale()

		elif command == "EQUALIZE":
			image.equalize()
		
		elif command == "UNBLUR":
			image.unblur()

		# Machine-Learning algorithms
		# SIFT (scale (zoom in/out), rotation, illumination, perspective)

		elif command == "SIFT":
			image.sift()

		elif command == "RANSAC":
			image.ransac()

		elif command == "PALM":
			image.find_palm_lines()
		
		elif command == "DETECT_FACES":
			image.detect_face()

		# UTILS:

		elif command == "SHOW":
			image.show()

		elif command == "SAVE":
			outputFilename = input("Enter filename to save: ")
			image.save(outputFilename)
		
		elif command == "EXIT":
			print("Exiting the application.")
			break

		# OTHER:

		elif command == "BLEND":
			image.blend()

		# TODO: Anca, add more features

		else:
			print("Invalid command. Please try again.")

if __name__ == "__main__":
	main()
