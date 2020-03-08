# TrespassingDetection

Download all the files in the folder named "Complete folder to run the program". Have all these files in the same folder. Install and setup OpenVINO environment to run the program. In the send_mail.py file, put your email id as a parameter where the "main" function is called. 

Run the below mentioned command in the shell to obtain results.

python app.py -m pedestrian-detection-adas-0002.xml -ct 0.6 -c BLUE

As of now, you can pass a video file as input to detect trespassers. In real time, we can use camera to detect trespassers. Also, we can have an alarm system to alert security guards around premises.

# Obtained results

If a person is detected in a prohibited area, an alert mail is sent immediately to the concerned team with the image of the trespasser attached.
