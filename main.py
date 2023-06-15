import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from copy import copy, deepcopy
import json
import numpy as np
import utm
from sklearn.cluster import DBSCAN

# NECESSARY FUNCTION
def ConvertLatLonArrayToUTM( LatitudeArray, LongitudeArray ):
	'''
	Reference : https://gis.stackexchange.com/questions/413420/converting-utm-to-lat-long-with-pythons-utm
	'''
	utmEasting = []
	utmNorthing = []
	utmZoneNumber = []
	utmZoneChar = []
	for arrayIndex in range( len( LatitudeArray ) ):
		utmCoordinate = utm.from_latlon( LatitudeArray[ arrayIndex ], LongitudeArray[ arrayIndex ] )
		utmEasting.append( copy( utmCoordinate[ 0 ] ) )
		utmNorthing.append( copy( utmCoordinate[ 1 ] ) )
		utmZoneNumber.append( copy( utmCoordinate[ 2 ] ) )
		utmZoneChar.append( copy( utmCoordinate[ 3 ] ) )
	return utmEasting, utmNorthing

# USER INTERFACE FUNCTION
def plotData( ):
	## GET CURRENT DATA
	CurrentLongitudeData = EastingDataFrame[ time ]
	CurrentLatitudeData = NorthingDataFrame[ time ]
	CurrentCompassHeadingData = CompassHeadingDataFrame[ time ]
	CurrentLidarRangeMeter = np.array( json.loads( LidarRangeMeterDataFrame[ time ] ) )
	CurrentLidarAngleDegree = np.array( json.loads( LidarAngleDegreeDataFrame[ time ] ) )

	## CONVERT FRAME OF COMPASS AND LIDAR FOR PLOT
	CurrentRobotFrontOrientationData = np.rad2deg((np.arctan2(-np.sin( np.deg2rad(CurrentCompassHeadingData - 90)),
													np.cos( np.deg2rad( CurrentCompassHeadingData - 90 )))))
	CurrentLidarAngleRefGlobalFrame = CurrentRobotFrontOrientationData - CurrentLidarAngleDegree
	
	## PREPARE DATA TO PLOT
	# Prepare X, Y Coordinate
	VisualizeLongitudeArray = EastingDataFrame[ :time ]
	VisualizeLatitudeArray = NorthingDataFrame[ :time ]

	## Pre[are plot Orientation of front of robot
	ArrowLength = 0.5
	OrientationArrowAttributes = [CurrentLongitudeData, CurrentLatitudeData,
								  (np.cos(np.deg2rad(CurrentRobotFrontOrientationData)) * ArrowLength),
								  (np.sin(np.deg2rad(CurrentRobotFrontOrientationData)) * ArrowLength)]
		
	## Prepare plot Lidar Scan Points
	CurrentLidarScanPoint = [ CurrentLongitudeData + (CurrentLidarRangeMeter * np.cos(np.deg2rad(CurrentLidarAngleRefGlobalFrame))),
							CurrentLatitudeData + ( CurrentLidarRangeMeter * np.sin( np.deg2rad( CurrentLidarAngleRefGlobalFrame ))) ]


	## CLUSTERING SECTION
	'''
	DBSCAN Clustering : https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
	'''
	# Initialize the DBSCAN clustering model
	dbscanModel = DBSCAN()
	
	# Fit the model to the data
	CurrentLidarScanPointArray = np.array( CurrentLidarScanPoint ).T
	dbscanModel.fit( CurrentLidarScanPointArray )
	
	# Get the cluster labels
	clusterLabels = dbscanModel.labels_
	
	# Initialize Variable for save clustering data
	clusterXAxisList = []
	clusterYAxisList = []
	clusterlabelList = []
	
	subplot.clear()
	
	# convert clusterLabels list to the set
	uniqueLabels = set( clusterLabels )
	for labelNumber in uniqueLabels:
		# Get X,Y point coordinate of each cluster
		clusterXAxis = CurrentLidarScanPoint[ 0 ][clusterLabels == labelNumber]
		clusterYAxis = CurrentLidarScanPoint[ 1 ][clusterLabels == labelNumber]
		
		# just save clustering data to use in linear interpolation
		clusterXAxisList.append( clusterXAxis )
		clusterYAxisList.append( clusterYAxis )
		clusterlabelList.append( labelNumber )
		
		# for plot by cluster
		if showClusterFlag == True:
			subplot.scatter(clusterXAxis, clusterYAxis, s=10, label=f'Cluster {labelNumber}')
	
	# for plot with out cluster
	if showClusterFlag == False:
		subplot.plot(CurrentLidarScanPoint[ 0 ], CurrentLidarScanPoint[1], '.k', label="LidarScan" )
		subplot.legend( loc="upper right" )


	## LINEAR FITTING SECTION
	if linearFitFlag == True:
		'''
			linear fiting: https://www.pythonpool.com/numpy-polyfit/
		'''
		for labelIndex in range( len( clusterlabelList ) ):
			if clusterlabelList[ labelIndex ] != -1:
				if len( clusterXAxisList[ labelIndex ] ) >= 2:
					lineCoef = np.polyfit( clusterXAxisList[ labelIndex ],  clusterYAxisList[ labelIndex ], 1 )
					poly1d_fn = np.poly1d( lineCoef )
					subplot.plot(  clusterXAxisList[ labelIndex ], poly1d_fn( clusterXAxisList[ labelIndex ] ),  'r'  )
	
	subplot.title.set_text( 'GPS & LIDAR VISUALIZER 2000' )
	subplot.set_xlabel( 'UTM_EASTING ( meters )' )
	subplot.set_ylabel( 'UTM_NORTHING ( meters )' )
	subplot.axis( GpsVisualizeBoundary )
	subplot.plot( VisualizeLongitudeArray, VisualizeLatitudeArray, 'g-', label="Path" )
	subplot.legend( loc="upper right" )
	subplot.plot( VisualizeLongitudeArray, VisualizeLatitudeArray, 'bo', label="Position", markersize=2  )
	subplot.legend( loc="upper right" )
	subplot.arrow( OrientationArrowAttributes[ 0 ], OrientationArrowAttributes[ 1 ],
			   	OrientationArrowAttributes[ 2 ], OrientationArrowAttributes[ 3 ], color = 'y',
				label="Orientation" )
	subplot.legend( loc="upper right" )
	canvas.draw()

def updateLoop():
	global time, playFlag
	if playFlag == True:
		plotData( )
		if time < maxTime:
			time += 1
		elif time >= maxTime:
			playFlag = False
	FrameSelectorSliderBar.set(time)
	TimeShowLable[ "text" ] = f"{ time }/{ maxTime } sec"
	window.after( int( 1000/speedFactor ), updateLoop )

def pauseButtonCallback():
	global playFlag
	playFlag = False

def playButtonCallback():
	global playFlag
	playFlag = True

def showClusterButtonCallback():
	global showClusterFlag
	if showClusterFlag:
		showClusterFlag = False
	else:
		showClusterFlag = True
	plotData( )

def LinearFitButtonCallback():
	global linearFitFlag
	if linearFitFlag:
		linearFitFlag = False
	else:
		linearFitFlag = True
	plotData( )

def frameSliderBarCallback(value):
	global time
	time = int( value )
	plotData( )

def UpdateSpeedSliderBarCallback(value):
	global speedFactor
	speedFactor = int( value )


## CONFIG
WINDOWWIDTH = 720
WINDOWHEIGHT = 576
PLOTEXPANDBOUNDFACTOR = 5

## UPLOAD FILES
# Read CSV files
GpsDataFrame = pd.read_csv( 'gpsPlus_20230612164330.csv' )
LidarDataFrame = pd.read_csv( 'ydlidar_20230612164330.csv' )

## GET DATA
# Time Reference DataFrame
TimeDataFrame = list( GpsDataFrame[ 'time_sec' ] )

# Extract latitude( metre ), longitude( metre ), RobotOrientation( Degree ) values from the GpsDataFrame
LongitudeDataFrame = np.array( GpsDataFrame[ 'gps_recentLongitudeE' ] )
LatitudeDataFrame = np.array( GpsDataFrame[ 'gps_recentLatitudeN' ] )
CompassHeadingDataFrame = np.array( GpsDataFrame[ 'compass_heading_degs' ] )
VehicleHeadingDataFrame = np.array( GpsDataFrame[ 'gps_vehicleHeading_degs' ] )

# Extract lidar_angle_degree ( degree ), lidar_range_meter( metre ) attributes
LidarAngleDegreeDataFrame = np.array( LidarDataFrame[ 'lidar_angle_degree' ] )
LidarRangeMeterDataFrame = np.array( LidarDataFrame[ 'lidar_range_meter' ] )

## PRE-PROCESSING GPS COORDINATE
## Convert Longitude and Latitude to UTM Northing, Easting metre
EastingDataFrame, NorthingDataFrame = ConvertLatLonArrayToUTM( LatitudeDataFrame, LongitudeDataFrame )
## Offset Coordinate by set the start pont of robot as ( 0, 0 ) of Global Frame
EastingDataFrame -= EastingDataFrame[ 0 ]
NorthingDataFrame -= NorthingDataFrame[ 0 ]

## CONFIG PLOT BOUNDARY
GpsVisualizeBoundary = [ np.min( EastingDataFrame )-PLOTEXPANDBOUNDFACTOR, np.max( EastingDataFrame )+PLOTEXPANDBOUNDFACTOR,
						 np.min( NorthingDataFrame )-PLOTEXPANDBOUNDFACTOR, np.max( NorthingDataFrame )+PLOTEXPANDBOUNDFACTOR ]

## INITIALIZE VARIABLES
time = 0
minTime = 0
maxTime = len( TimeDataFrame )-1
playFlag = False
showClusterFlag = False
linearFitFlag = False
speedFactor = 10

### UI PARTS
# Create window with Tkinter
window = tk.Tk()
window.title( "GPS & LIDAR VISUALIZER 2000" )
window.resizable( width=False, height=False )
window.geometry( f"{ WINDOWWIDTH }x{ WINDOWHEIGHT }")

# Create a Matplotlib figure
figure = Figure( figsize=( 5, 4 ), dpi=100 )
subplot = figure.add_subplot( 1, 1, 1 )

# Create a Tkinter canvas that contains the Matplotlib figure
canvas = FigureCanvasTkAgg( figure, master=window )
canvas.draw()
canvas.get_tk_widget().place( x=0, y=50, width = WINDOWWIDTH, height =WINDOWHEIGHT - 50 )
plotData()

# Create a button to trigger the plot function
PlayButton = tk.Button( window, text="Play", command=playButtonCallback )
PlayButton.place( x=0, y=0, width = 50, height = 50 )

PauseButton = tk.Button( window, text="Pause", command=pauseButtonCallback )
PauseButton.place( x=50, y=0, width = 50, height = 50 )

FrameSliderLabel = tk.Label( master=window, text="Frame\nNumber" )
FrameSliderLabel.place( x =100, y = 0, width = 50, height = 50 )

FrameSelectorSliderBar = tk.Scale( window, from_=minTime, to=maxTime, orient=tk.HORIZONTAL, command=frameSliderBarCallback )
FrameSelectorSliderBar.set( time )  # Initial value
FrameSelectorSliderBar.place( x=150, y=0, width =200, height = 50 )

UpdateSpeedSliderLabel = tk.Label( master=window, text="Update\nSpeed" )
UpdateSpeedSliderLabel.place( x = 350, y = 0, width = 50, height = 50 )

UpdateSpeedSliderBar = tk.Scale( window, from_=1, to=10, orient=tk.HORIZONTAL, command=UpdateSpeedSliderBarCallback )
UpdateSpeedSliderBar.set( 10 )  # Initial value
UpdateSpeedSliderBar.place(x=400, y=0, width =100, height = 50)

ShowClusterButton = tk.Button( window, text="ShowCluster", command=showClusterButtonCallback )
ShowClusterButton.place( x=500, y=0, width = 100, height = 25 )

LinearFitButton = tk.Button( window, text="ShowLineFit", command=LinearFitButtonCallback )
LinearFitButton.place( x=500, y=25, width = 100, height = 25 )

TimeShowLable = tk.Label( master=window, text=f"{ time }/{ maxTime } sec" )
TimeShowLable.place( x =WINDOWWIDTH - 100, y = 0, width = 100, height = 40 )


# Setup update loop
updateLoop()

# Run the Tkinter event loop
window.mainloop()
