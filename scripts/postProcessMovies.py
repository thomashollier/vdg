#!/usr/bin/env python3

import OpenImageIO as oiio
from OpenImageIO import ImageBuf, ImageBufAlgo
import numpy as np
import sys
import os

def procOnWhite(input_file, dry_run=False):
	"""
	Process premultiplied RGB with alpha channel manipulation and compositing.
	Alpha processing: chsum -> sigmoid contrast -> blur -> power
	"""
	if not dry_run:
		# Generate filenames
		alpha_file = input_file.replace('.png', '_alpha.png')
		output_file = input_file.replace('.png', '_procOnWhite.png')
		
		if not os.path.exists(input_file) or not os.path.exists(alpha_file):
			print(f"Error: Required files not found")
			return False
		
		print(f"Processing: {input_file}")
		
		# Load images as uint16
		rgb_img_temp = ImageBuf(input_file)
		spec = rgb_img_temp.spec()
		spec.set_format(oiio.UINT16)
		rgb_img = ImageBuf(spec)
		ImageBufAlgo.copy(rgb_img, rgb_img_temp)
		
		alpha_img_temp = ImageBuf(alpha_file)
		alpha_spec = alpha_img_temp.spec()
		alpha_spec.set_format(oiio.UINT16)
		alpha_img = ImageBuf(alpha_spec)
		ImageBufAlgo.copy(alpha_img, alpha_img_temp)
		
		# Extract RGB and red channel from alpha
		rgb_only = ImageBufAlgo.channels(rgb_img, (0, 1, 2))
		red_channel = ImageBufAlgo.channels(alpha_img, (0,))
		
		# Append red channel as alpha and unpremult
		rgba_img = ImageBufAlgo.channel_append(rgb_only, red_channel)
		rgba_img = ImageBufAlgo.channels(rgba_img, (0, 1, 2, 3), newchannelnames=("R", "G", "B", "A"))
		
		# Composite over white background
		background = ImageBuf(oiio.ImageSpec(spec.width, spec.height, 4, oiio.UINT16))
		ImageBufAlgo.fill(background, (1.0, 1.0, 1.0, 1.0))
		composited = ImageBufAlgo.over(rgba_img, background)
		
		# Extract RGB and save
		result_rgb = ImageBufAlgo.channels(composited, (0, 1, 2))
		final_spec = result_rgb.spec()
		final_spec.set_format(oiio.UINT16)
		result = ImageBuf(final_spec)
		ImageBufAlgo.copy(result, result_rgb)
		
		success = result.write(output_file, oiio.UINT16)
		if success:
			print(f"Saved: {output_file}")
		else:
			print(f"Error writing output: {result.geterror()}")
		
		return success
	else:
		print("dry run, no post process")


def procGamma(input_file, gamma=2.2, dry_run=False):
	"""
	Process premultiplied RGB with alpha channel manipulation and compositing.
	Alpha processing: chsum -> sigmoid contrast -> blur -> power
	"""
	if not dry_run:
		# Generate filenames
		alpha_file = input_file.replace('.png', '_alpha.png')
		output_file = input_file.replace('.png', '_procGamma.png')
		
		if not os.path.exists(input_file) or not os.path.exists(alpha_file):
			print(f"Error: Required files not found")
			return False
		
		print(f"Processing: {input_file}")
		
		# Load images as uint16
		rgb_img_temp = ImageBuf(input_file)
		spec = rgb_img_temp.spec()
		spec.set_format(oiio.UINT16)
		rgb_img = ImageBuf(spec)
		ImageBufAlgo.copy(rgb_img, rgb_img_temp)
		
		alpha_img_temp = ImageBuf(alpha_file)
		alpha_spec = alpha_img_temp.spec()
		alpha_spec.set_format(oiio.UINT16)
		alpha_img = ImageBuf(alpha_spec)
		ImageBufAlgo.copy(alpha_img, alpha_img_temp)
		
		# Extract RGB and red channel from alpha
		rgb_only = ImageBufAlgo.channels(rgb_img, (0, 1, 2))
		red_channel = ImageBufAlgo.channels(alpha_img, (0,))
		
		# Append red channel as alpha and unpremult
		rgba_img = ImageBufAlgo.channel_append(rgb_only, red_channel)
		rgba_img = ImageBufAlgo.channels(rgba_img, (0, 1, 2, 3), newchannelnames=("R", "G", "B", "A"))
		unpremult_img = ImageBufAlgo.unpremult(rgba_img)
		
		# Power
		processed_alpha = ImageBufAlgo.channel_sum(alpha_img, (1.0, 0.0, 0.0))
		processed_alpha = ImageBufAlgo.pow(processed_alpha, 1/gamma)
		pixels = processed_alpha.get_pixels(oiio.FLOAT)
		pixels = np.clip(pixels, 0.0, 1.0)
		p_spec = processed_alpha.spec()
		p_spec.set_format(oiio.UINT16)
		processed_alpha_final = ImageBuf(p_spec)
		processed_alpha_final.set_pixels(oiio.ROI(0, p_spec.width, 0, p_spec.height, 0, 1, 0, p_spec.nchannels), pixels)
		
		# Swap alpha and premultiply
		rgb_straight = ImageBufAlgo.channels(unpremult_img, (0, 1, 2))
		rgba_with_processed = ImageBufAlgo.channel_append(rgb_straight, processed_alpha_final)
		rgba_with_processed = ImageBufAlgo.channels(rgba_with_processed, (0, 1, 2, 3), 
													newchannelnames=("R", "G", "B", "A"))
		result_premult = ImageBufAlgo.premult(rgba_with_processed)
		
		# Composite over white background
		background = ImageBuf(oiio.ImageSpec(spec.width, spec.height, 4, oiio.UINT16))
		ImageBufAlgo.fill(background, (1.0, 1.0, 1.0, 1.0))
		composited = ImageBufAlgo.over(result_premult, background)
		
		# Extract RGB and save
		result_rgb = ImageBufAlgo.channels(composited, (0, 1, 2))
		final_spec = result_rgb.spec()
		final_spec.set_format(oiio.UINT16)
		result = ImageBuf(final_spec)
		ImageBufAlgo.copy(result, result_rgb)
		
		success = result.write(output_file, oiio.UINT16)
		if success:
			print(f"Saved: {output_file}")
		else:
			print(f"Error writing output: {result.geterror()}")
		
		return success
	else:
		print("dry run, no post process")

def proc00(input_file, dry_run=False):
	"""
	Process premultiplied RGB with alpha channel manipulation and compositing.
	Alpha processing: chsum -> sigmoid contrast -> blur -> power
	"""
	if not dry_run:
		# Generate filenames
		alpha_file = input_file.replace('.png', '_alpha.png')
		output_file = input_file.replace('.png', '_proc00.png')
		
		if not os.path.exists(input_file) or not os.path.exists(alpha_file):
			print(f"Error: Required files not found")
			return False
		
		print(f"Processing: {input_file}")
		
		# Load images as uint16
		rgb_img_temp = ImageBuf(input_file)
		spec = rgb_img_temp.spec()
		spec.set_format(oiio.UINT16)
		rgb_img = ImageBuf(spec)
		ImageBufAlgo.copy(rgb_img, rgb_img_temp)
		
		alpha_img_temp = ImageBuf(alpha_file)
		alpha_spec = alpha_img_temp.spec()
		alpha_spec.set_format(oiio.UINT16)
		alpha_img = ImageBuf(alpha_spec)
		ImageBufAlgo.copy(alpha_img, alpha_img_temp)
		
		# Extract RGB and red channel from alpha
		rgb_only = ImageBufAlgo.channels(rgb_img, (0, 1, 2))
		red_channel = ImageBufAlgo.channels(alpha_img, (0,))
		
		# Append red channel as alpha and unpremult
		rgba_img = ImageBufAlgo.channel_append(rgb_only, red_channel)
		rgba_img = ImageBufAlgo.channels(rgba_img, (0, 1, 2, 3), newchannelnames=("R", "G", "B", "A"))
		unpremult_img = ImageBufAlgo.unpremult(rgba_img)
		
		# Process alpha: chsum -> sigmoid contrast -> blur -> power
		processed_alpha = ImageBufAlgo.channel_sum(alpha_img, (1.0, 0.0, 0.0))
		pixels = processed_alpha.get_pixels(oiio.FLOAT)
		pixels = np.clip(pixels, 0.0, 1.0)
		p_spec = processed_alpha.spec()
		processed_alpha = ImageBuf(p_spec)
		processed_alpha.set_pixels(oiio.ROI(0, p_spec.width, 0, p_spec.height, 0, 1, 0, p_spec.nchannels), pixels)
		
		# Sigmoid contrast
		pixels = processed_alpha.get_pixels(oiio.FLOAT)
		scontrast = 60.0
		sthresh = 0.0015
		exponent = -scontrast * (pixels - sthresh)
		exponent = np.clip(exponent, -88, 88)
		pixels = 1.0 / (1.0 + np.exp(exponent))
		pixels = np.clip(pixels, 0.0, 1.0)
		p_spec = processed_alpha.spec()
		processed_alpha = ImageBuf(p_spec)
		processed_alpha.set_pixels(oiio.ROI(0, p_spec.width, 0, p_spec.height, 0, 1, 0, p_spec.nchannels), pixels)
		
		# Blur
		kernel = ImageBuf(oiio.ImageSpec(5, 5, 1, oiio.FLOAT))
		ImageBufAlgo.make_kernel(kernel, "gaussian", 5.0, 5.0)
		processed_alpha = ImageBufAlgo.convolve(processed_alpha, kernel)
		pixels = processed_alpha.get_pixels(oiio.FLOAT)
		pixels = np.clip(pixels, 0.0, 1.0)
		p_spec = processed_alpha.spec()
		processed_alpha = ImageBuf(p_spec)
		processed_alpha.set_pixels(oiio.ROI(0, p_spec.width, 0, p_spec.height, 0, 1, 0, p_spec.nchannels), pixels)
		
		# Power
		processed_alpha = ImageBufAlgo.pow(processed_alpha, 8.0)
		pixels = processed_alpha.get_pixels(oiio.FLOAT)
		pixels = np.clip(pixels, 0.0, 1.0)
		p_spec = processed_alpha.spec()
		p_spec.set_format(oiio.UINT16)
		processed_alpha_final = ImageBuf(p_spec)
		processed_alpha_final.set_pixels(oiio.ROI(0, p_spec.width, 0, p_spec.height, 0, 1, 0, p_spec.nchannels), pixels)
		
		# Swap alpha and premultiply
		rgb_straight = ImageBufAlgo.channels(unpremult_img, (0, 1, 2))
		rgba_with_processed = ImageBufAlgo.channel_append(rgb_straight, processed_alpha_final)
		rgba_with_processed = ImageBufAlgo.channels(rgba_with_processed, (0, 1, 2, 3), 
													newchannelnames=("R", "G", "B", "A"))
		result_premult = ImageBufAlgo.premult(rgba_with_processed)
		
		# Composite over white background
		background = ImageBuf(oiio.ImageSpec(spec.width, spec.height, 4, oiio.UINT16))
		ImageBufAlgo.fill(background, (1.0, 1.0, 1.0, 1.0))
		composited = ImageBufAlgo.over(result_premult, background)
		
		# Extract RGB and save
		result_rgb = ImageBufAlgo.channels(composited, (0, 1, 2))
		final_spec = result_rgb.spec()
		final_spec.set_format(oiio.UINT16)
		result = ImageBuf(final_spec)
		ImageBufAlgo.copy(result, result_rgb)
		
		success = result.write(output_file, oiio.UINT16)
		if success:
			print(f"Saved: {output_file}")
		else:
			print(f"Error writing output: {result.geterror()}")
		
		return success
	else:
		print("dry run, no post process")

def trim(input_file, output_file=None, fuzz_percent=1, verbose=False):
	"""
	Trim an image with fuzz tolerance.
	
	This replicates the trimming part of the bash doTrimB function:
	magick $i -fuzz 1% -trim +repage $io
	
	Args:
		input_file: Path to input image file (e.g., 'image.png')
		output_file: Path to output file. If None, generates from input (e.g., 'image_trimb.png')
		fuzz_percent: Fuzz tolerance percentage for trimming (default: 1)
		verbose: Print command and output
		
	Returns:
		Path to output file if successful, None otherwise
	"""
	if not os.path.exists(input_file):
		print(f"Warning: Input file {input_file} not found")
		return None
	
	# Generate output filename if not provided
	if output_file is None:
		base, ext = os.path.splitext(input_file)
		output_file = f"{base}_trimb{ext}"
	
	if verbose:
		print(f"Trimming image: {input_file} -> {output_file}")
	
	try:
		# Load the image
		img = ImageBuf(input_file)
		if img.has_error:
			print(f"Error loading image: {img.geterror()}")
			return None
		
		# Get original dimensions
		spec = img.spec()
		orig_width = spec.width
		orig_height = spec.height
		
		# Convert fuzz_percent to threshold (0-1 range)
		fuzz_threshold = fuzz_percent / 100.0
		
		# Get pixels as numpy array (much faster than pixel-by-pixel access)
		pixels = img.get_pixels(oiio.FLOAT)
		
		# Sample all four corners to determine background color
		corners = [
			pixels[0, 0, :3],
			pixels[0, -1, :3],
			pixels[-1, 0, :3],
			pixels[-1, -1, :3]
		]
		# Use the average of corners as background reference
		corner_color = np.mean(corners, axis=0)
		
		if verbose:
			print(f"  Detected background color: {corner_color}")
		
		# Vectorized difference calculation
		# Calculate max difference across RGB channels for each pixel
		diff = np.max(np.abs(pixels[:, :, :3] - corner_color), axis=2)
		
		# Find pixels that differ from background
		content_mask = diff > fuzz_threshold
		
		if verbose:
			content_pixels = np.sum(content_mask)
			total_pixels = content_mask.size
			print(f"  Content pixels: {content_pixels}/{total_pixels} ({100*content_pixels/total_pixels:.1f}%)")
		
		# Find bounding box using numpy operations (much faster than loops)
		rows_with_content = np.any(content_mask, axis=1)
		cols_with_content = np.any(content_mask, axis=0)
		
		row_indices = np.where(rows_with_content)[0]
		col_indices = np.where(cols_with_content)[0]
		
		if len(row_indices) == 0 or len(col_indices) == 0:
			if verbose:
				print("Warning: No content found to trim, using original image")
			min_y, max_y = 0, orig_height - 1
			min_x, max_x = 0, orig_width - 1
		else:
			min_y = int(row_indices[0])
			max_y = int(row_indices[-1])
			min_x = int(col_indices[0])
			max_x = int(col_indices[-1])
		
		# Crop to the bounding box
		crop_width = max_x - min_x + 1
		crop_height = max_y - min_y + 1
		
		if verbose:
			print(f"  Original: {orig_width}x{orig_height}")
			print(f"  Crop region: ({min_x},{min_y}) to ({max_x},{max_y})")
			print(f"  Trimmed to: {crop_width}x{crop_height}")
		
		cropped = ImageBufAlgo.cut(img, oiio.ROI(min_x, max_x + 1, min_y, max_y + 1))
		
		# Write output
		success = cropped.write(output_file)
		if success:
			if verbose:
				print(f"  Successfully created: {output_file} ({crop_width}x{crop_height})")
			return output_file
		else:
			print(f"Error writing output: {cropped.geterror()}")
			return None
			
	except Exception as e:
		print(f"Error processing image: {e}")
		import traceback
		traceback.print_exc()
		return None

def trim_and_scale(input_file, output_file=None, fuzz_percent=1, scale_x=1.0, scale_y=1.0, background_color="auto", verbose=False):
	"""
	Trim an image with fuzz tolerance and then scale it.
	
	Args:
		input_file: Path to input image file (e.g., 'image.png')
		output_file: Path to output file. If None, generates from input (e.g., 'image_trimb_scaled.png')
		fuzz_percent: Fuzz tolerance percentage for trimming (default: 1)
		scale_x: X scale factor (default: 1.0)
		scale_y: Y scale factor (default: 1.0)
		background_color: Background color as "(r,g,b)" string with 0-1 range or "auto" (default: "auto")
		verbose: Print command and output
		
	Returns:
		Path to output file if successful, None otherwise
	"""
	if not os.path.exists(input_file):
		print(f"Warning: Input file {input_file} not found")
		return None
	
	# Generate output filename if not provided
	if output_file is None:
		base, ext = os.path.splitext(input_file)
		output_file = f"{base}_trimb_scaled{ext}"
	
	if verbose:
		print(f"Trimming and scaling image: {input_file} -> {output_file}")
		print(f"  Scale factors: X={scale_x}, Y={scale_y}")
	
	try:
		# Load the image
		img = ImageBuf(input_file)
		if img.has_error:
			print(f"Error loading image: {img.geterror()}")
			return None
		
		# Get original dimensions
		spec = img.spec()
		orig_width = spec.width
		orig_height = spec.height
		
		# Convert fuzz_percent to threshold (0-1 range)
		fuzz_threshold = fuzz_percent / 100.0
		
		# Get pixels as numpy array (much faster than pixel-by-pixel access)
		pixels = img.get_pixels(oiio.FLOAT)
		
		# Parse background color
		if background_color.lower() == "auto":
			# Sample all four corners to determine background color
			corners = [
				pixels[0, 0, :3],
				pixels[0, -1, :3],
				pixels[-1, 0, :3],
				pixels[-1, -1, :3]
			]
			# Use the average of corners as background reference
			corner_color = np.mean(corners, axis=0)
		else:
			# Parse "(r,g,b)" format
			try:
				# Remove parentheses and split by comma
				color_str = background_color.strip("()").replace(" ", "")
				r, g, b = map(float, color_str.split(","))
				corner_color = np.array([r, g, b])
			except Exception as e:
				print(f"Error parsing background_color '{background_color}': {e}")
				print("Expected format: '(r,g,b)' with values 0-1, e.g. '(0,0,0)' or 'auto'")
				return None
		
		if verbose:
			print(f"  Background color: {corner_color}")
		
		# Vectorized difference calculation
		# Calculate max difference across RGB channels for each pixel
		diff = np.max(np.abs(pixels[:, :, :3] - corner_color), axis=2)
		
		# Find pixels that differ from background
		content_mask = diff > fuzz_threshold
		
		if verbose:
			content_pixels = np.sum(content_mask)
			total_pixels = content_mask.size
			print(f"  Content pixels: {content_pixels}/{total_pixels} ({100*content_pixels/total_pixels:.1f}%)")
		
		# Find bounding box using numpy operations (much faster than loops)
		rows_with_content = np.any(content_mask, axis=1)
		cols_with_content = np.any(content_mask, axis=0)
		
		row_indices = np.where(rows_with_content)[0]
		col_indices = np.where(cols_with_content)[0]
		
		if len(row_indices) == 0 or len(col_indices) == 0:
			if verbose:
				print("Warning: No content found to trim, using original image")
			min_y, max_y = 0, orig_height - 1
			min_x, max_x = 0, orig_width - 1
		else:
			min_y = int(row_indices[0])
			max_y = int(row_indices[-1])
			min_x = int(col_indices[0])
			max_x = int(col_indices[-1])
		
		# Crop to the bounding box
		crop_width = max_x - min_x + 1
		crop_height = max_y - min_y + 1
		
		if verbose:
			print(f"  Original: {orig_width}x{orig_height}")
			print(f"  Crop region: ({min_x},{min_y}) to ({max_x},{max_y})")
			print(f"  Trimmed to: {crop_width}x{crop_height}")
		
		cropped = ImageBufAlgo.cut(img, oiio.ROI(min_x, max_x + 1, min_y, max_y + 1))
		
		# Calculate new dimensions after scaling
		new_width = int(crop_width * scale_x)
		new_height = int(crop_height * scale_y)
		
		if verbose:
			print(f"  Scaling to: {new_width}x{new_height}")
			if scale_x != scale_y:
				print(f"  Note: Aspect ratio will change (X:{scale_x}, Y:{scale_y})")
		
		# Get the spec to know how many channels we have
		cropped_spec = cropped.spec()
		
		# Use the ROI-based resize approach with blackman-harris filter for high quality
		roi = oiio.ROI(0, new_width, 0, new_height, 0, 1, 0, cropped_spec.nchannels)
		scaled = ImageBufAlgo.resize(cropped, filtername="blackman-harris", roi=roi)
		
		if scaled.has_error:
			print(f"Error during resize: {scaled.geterror()}")
			return None
		
		# Write output
		success = scaled.write(output_file)
		if success:
			if verbose:
				print(f"  Successfully created: {output_file} ({new_width}x{new_height})")
			return output_file
		else:
			print(f"Error writing output: {scaled.geterror()}")
			return None
			
	except Exception as e:
		print(f"Error processing image: {e}")
		import traceback
		traceback.print_exc()
		return None

def main():
	if len(sys.argv) < 3:
		print("Usage: python postProcessMovies.py <function_name> <input_file> [output_file] [options]")
		print("\nAvailable functions:")
		print("  proc00 <input_file> [dry_run]")
		print("  procGamma <input_file> [gamma] [dry_run]")
		print("  procOnWhite <input_file> [dry_run]")
		print("  trim <input_file> [output_file] [fuzz_percent] [verbose]")
		print("  trim_and_scale <input_file> [output_file] [fuzz_percent] [scale_x] [scale_y] [background_color] [verbose]")
		print("\nExamples:")
		print("  python postProcessMovies.py proc00 input.png")
		print("  python postProcessMovies.py procGamma input.png 2.2")
		print("  python postProcessMovies.py procOnWhite input.png")
		print("  python postProcessMovies.py trim input.png output.png 1 True")
		print("  python postProcessMovies.py trim input.png")
		print("  python postProcessMovies.py trim_and_scale input.png output.png 1 0.5 0.5 auto True")
		print("  python postProcessMovies.py trim_and_scale input.png None 1 2.0 2.0 '(0,0,0)'")
		print("  python postProcessMovies.py trim_and_scale input.png None 1 1.0 0.5 '(1,1,1)' True")
		sys.exit(1)
	
	function_name = sys.argv[1]
	input_file = sys.argv[2]
	
	success = False
	
	try:
		if function_name == "proc00":
			dry_run = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
			success = proc00(input_file, dry_run=dry_run)
			
		elif function_name == "procGamma":
			gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 2.2
			dry_run = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
			success = procGamma(input_file, gamma=gamma, dry_run=dry_run)
			
		elif function_name == "procOnWhite":
			dry_run = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
			success = procOnWhite(input_file, dry_run=dry_run)
			
		elif function_name == "trim":
			output_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() != 'none' else None
			fuzz_percent = int(sys.argv[4]) if len(sys.argv) > 4 else 1
			verbose = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False
			
			result = trim(input_file, output_file=output_file, 
			              fuzz_percent=fuzz_percent, verbose=verbose)
			success = result is not None
			
		elif function_name == "trim_and_scale":
			output_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].lower() != 'none' else None
			fuzz_percent = int(sys.argv[4]) if len(sys.argv) > 4 else 1
			scale_x = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
			scale_y = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
			background_color = sys.argv[7] if len(sys.argv) > 7 else "auto"
			verbose = sys.argv[8].lower() == 'true' if len(sys.argv) > 8 else False
			
			result = trim_and_scale(input_file, output_file=output_file, 
			                        fuzz_percent=fuzz_percent, scale_x=scale_x, 
			                        scale_y=scale_y, background_color=background_color,
			                        verbose=verbose)
			success = result is not None
			
		else:
			print(f"Error: Unknown function '{function_name}'")
			print("Available functions: proc00, procGamma, procOnWhite, trim, trim_and_scale")
			sys.exit(1)
			
	except Exception as e:
		print(f"Error executing {function_name}: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
	
	sys.exit(0 if success else 1)


if __name__ == "__main__":
	main()
