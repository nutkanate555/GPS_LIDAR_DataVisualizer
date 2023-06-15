# RainaAssignment1
Assignment for software developer candidate

## Requirement
```
pip install tk
pip install utm
pip install matplotlib
pip install numpy
pip install pandas
pip install -U scikit-learn
```
## How to use?
1. Put the files ( main, 2 data files ) in the same folder then run.
2. This program has 4 buttons.
   - Play Button : For play
   - Pause Button : For pause
   - Show Cluster Button : For label the lidar scan points as cluster using dbscan clustering
   - Show Line Fit : For visualize line that that fit to the lidar scan points
3. And we have 2 Slider Bar.
   - Frame Number Sliderbar : For select the Specific time frame to visualize.
   - Update Speed : Visualize speed can be adjust from 1-10 times

## How this program do?
1. Convert lat-lon to UTM (metre) -> This section will use library named "utm" to convert latlon by using utm.from_latlon() function.
From this fuction, we will get output: Easting ( metre ), Northing ( metre ), Zone*  
*Note : I ignore the zone because of this dataset the robot not change the UTM Zone
2. Do some rotation on sensors ( Compass, LidarAngle ) to able to use.
3. Using dbscan clustering model ( from scikit learn ) to cluster the lidar scan points.
4. Using numpy polyfit function ( 1st order ) to fit each cluster , exclude : cluster label -1
5. Finish !!

## Demo
