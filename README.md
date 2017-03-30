Cognitive Human Robot Interaction.

Formulate a robot model that can adjust its behavior to interact and collaborate effectively with a
human whose own behavior and performance are subject to unpredictable changes like mood swings,
changes of emotional states, boredom etc.


Read Goal Statement for more details...


Scenario: There are two locations, one where a bunch of raw materials 
are sitting and another where employee is working. Employee examines raw materials while a robot has to pick a raw material and place it next to employee. Please note that this is a repetitive task.

Objective: To predict when the employee is going to ask for the next material and plan accordingly (generate energy efficient trajectory).

Implementation

A video recording of employee asking for raw material (We used an extending human hand as a sign for raw material).

python request_detector.py -v <video-filename.mp4> (Timestamp of detection of Hand)
python request_detector_integrated.py -v <video-filename.mp4>  (Prediction of next reuest)

  

