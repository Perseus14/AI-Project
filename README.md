**Cognitive Human Robot Interaction.**

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

**Cognitive Multi Human Robot Interaction.**

Based on previous task times of each human in the same environment mentioned above, Future task times for multiple humans have been predicted and requests are scheduled.

DATA GENERATION AND MODEL FITTING:

python3 prediction.py

Once you execute the above command
1. Data is generated and the graph of data appears
2. After you close it, ACF and PACF plots of the data are generated (Use these to determine p,d,q,P,D,Q) 
3. After you close the plots enter the parametre values and enter S as 12.
4. Then the model that has been fit to the data will appear.



MULTIPLE HUMAN SCHEDULING SIMULATION:
python3 sim_multi_rand_FCFS.py 
python3 sim_multi_rand_SJF.py

These command generate the graph os jobs done vs time for each of the human using the scheduling policy.



