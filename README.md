# ParkingSpotDetectorDemo

## Overview

This project demonstrates the detection of cars in predefined parking spots using a YOLO-based object detection model. It provides an efficient interface to define, edit, and detect cars in parking spots on video footage, even from challenging camera angles and distances.

### Key Workflow

The workflow consists of three main steps:

1. **Drawing the parking spots**:  
   `draw-spots.py` - Manually define the regions on the video where parking spots are located by drawing trapezoids around each spot.
   
2. **Editing spots (if necessary)**:  
   `edit-spots.py` - Modify or adjust the parking spots after initial setup if needed. Simply drag and reshape the spots in case of changes to parking areas.

3. **Detecting cars in parking spots**:  
   Run `main.py` to detect car presence and track the occupancy of the defined parking spots.

## Installation

To set up the environment, install the required libraries by running:

```bash
pip install -r requirements.txt
