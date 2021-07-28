# Sketchizer

## Overview

Sketchizer is a program that enables you to turn an image into a hand drawing-like image in .svg format, using **python** and **openCV**. 

The app's purpose is to quickly generate a prototype that could be further manually edited using a .svg editor tool; or just to directly transform an image into a simpler outline.

### Requirements/venv



### How to use

When started, Sketchizer opens two windows:

A Test Window, where almost all the parameters of the image processing functions used are tunable.

A little GUI, which allows to select the **input** image and the **brush** used. It also has a button to show all the others processing parameters and a button to open the updated .svg file using the default web browser and render a hand-drawing animation.

## Implementation details

### Main.py

The main program essentially initializes all the parameters of the image processing functions, and sets up the interface.

Here all the **track bars** of the image windows are created, and the **refresh()** function, which re-runs all the processing with the current parameters, is called as soon as one slider is moved.

The window with the buttons is created using the library **tkinter** for python; the buttons switch some other parameters accessing lists in a cyclic way.

### Analyzer.py

This module contains all the image processing functions. Here are the most important.

#### **`find_lines(edges_img)`**

This function's purpose is to store each separate line in different data structures. Of course, the pixels in the line have to be stored in a sequential order to avoid a "back and forth" effect in the final drawing. 
To do so, a region growing-like algorithm has been implemented: a random pixel of an edge is selected and all the neighbors are explored in both directions. To assure the order, the pixels are appended or pushed in the list basing of the explored direction.

**`edges_img`** is a binary image containing the result of a Canny edge detection filter on the original image.

**`returns`** the edge detection image, where each separate line has been colored for debugging purposes.
**`returns`** a dictionary containing a key for each separate line. Each key is associated to an ordered list of pixels (coordinates in a tuple) of the corresponding line.

#### **`sample(lines, step)`**

**`lines`** is the dictionary containing all the lines.
**`step`** is the sampling step.

**`returns`** another dictionary with less pixels.

#### **`compute_derivatives(sampled)`**

**`sampled`** is the sampled dictionary containing less pixels for each line.

**`returns`** a dictionary containing, for each line, a tuple with the coordinates of the pixel and the slope associated to it. The slope is actually computed as the **arctangent** of `Δy/Δx`, and then divided by `π` in order to map in a range between -1 and +1.

#### **`find_flexes(derivatives, sampled, corner_thresh, closures, max_chances=2)`**

This function detects the flexes of the lines: they can be angular (corners) or points where the line changes direction (local maximum or minimum of the slope; in other terms, where the second derivative is zero).

**`derivatives`** is the dictionary containing the slope values for each sample of each line.
**`sampled`** is the sampled dictionary containing less pixels for each line.
**`corner_thresh`** is the threshold for detecting the corners.
**`closures`** is a list containing the keys of the closed lines. A corner is added at the start and at the end if the line is open.
**`max_chances`** represents how many consecutive slopes have to keep the changed direction before confirming the flex. Using zero, the flexes are confirmed just after finding a change in the direction; using a greater number is useful to reduce the noise. Usually 0 to 2 are good choices.

**`returns`** a dictionary containing, for each line, a tuple with the coordinates  of the flex, a flag to specify if the flex is a corner,  the new slope and the old one. These values will be useful later to set up the handles of the *polybezier* curves.

#### **`detect_shapes(lines, closures, approx_factor=0.9)`**

This function leverages openCV methods to detect basic shapes. An empty image is created and the single line is drawn in it; then the image is blurred and the `cv2.HoughCircles` function is called to check if it is a circle. To control the approximation level, the *HOUGH_GRADIENT* version is used, which accepts a "perfectness" threshold for the circles to detect. If no circles are detected, the coordinates of pixels are passed to the `cv2.approxPolyDP` function which simplifies it with less points: the number of points is checked to recognize the polygon.

**`lines`** is the dictionary containing all the lines.
**`closures`** is a list containing the keys of the closed lines.
**`approx_factor`** is the factor which determines the maximum number of points to use when approximating the line with `cv2.approxPolyDP` function. It also multiplies the "perfectness" threshold of `cv2.HoughCircles`. In general, it determines how much the lines are approximated to basic shapes (polygons and circles).

**`returns`** a dictionary containing, for each approximated line key, the name of the shape and the data to draw them.

### Test.py

This module contains all the debugging functions, which show the effects of the processing. Here are the most important.

#### **`test_flexes(lines_img, sampled, flexes, show_samples=False)`**

This function draws all the flexes as circles (in green, red for corners) on the input image. If the flag is set to true, it draws all the samples too (in white).

**`lines_img`** is the edge detection image, where each separate line has been colored for debugging purposes.
**`sampled`** is the sampled dictionary containing less pixels for each line.
**`flexes`** a dictionary containing, for each line, a tuple with the coordinates  of the flex, a flag to specify if the flex is a corner,  the new slope and the old one.
**`show_samples`** a flag to define whether the samples should be drawn or not. Default is fault.

**`returns`** The debug image with the flexes.

#### **`test_shapes(img, lines, shapes)`**

This function writes the name of the detected shape  on the input image, using the color of the line.

**`img`** is the input image.
**`lines`** is the dictionary containing all the lines.
**`shapes`** a dictionary containing, for each approximated line key, the name of the shape and the data to draw them.

**`returns`** The debug image with the texts of the detected shapes.

#### **`test_drawing()`**

This function reads the .svg output from the /output folder and writes a .html (animation.html) file which is associated with a .css stylesheet and a .js script. These files defines the properties and the filters of the paths in order to render a real-time animation of the output being "drawn".

### Vectorizer.py

This module contains functions for creating the .svg file. Here is the main one, which leverages the data about flexes, corners and slopes.

#### **`vectorize_flexes(img_x, img_y, flexes, closures, shapes, filter='pencil')`**

This function generates the output .svg file. For each line, the ordered nodes are added to the path and the handles are computed using the slope before and after each node. If the node is not angular, one of the handles is just the mirror of the previous one: this way the interpolation will be smoother.

**`img_x`** is the edge detection image, where each separate line has been colored for debugging purposes.
**`img_y`** is the sampled dictionary containing less pixels for each line.
**`flexes`** is a dictionary containing, for each line, a tuple with the coordinates  of the flex, a flag to specify if the flex is a corner,  the new slope and the old one.
**`closures`** is a list containing the keys of the closed lines. A corner is added at the start and at the end if the line is open.
**`shapes`** is a dictionary containing, for each approximated line key, the name of the shape and the data to draw them.
**`filter`** is a string which specifies the id of the filter (brush) to draw the paths and the shape with.

**`returns`** The final .svg file.

