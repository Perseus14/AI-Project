import matplotlib.pyplot as plt
import random
import sys

#global n
MAX_TIME = 100000
INT_MAX = 1000000000
#n =0
human_db = {}
r_curr_req_serving = -1
r_request = {}
r_free_status = True
r_pt = 10
mu = []
var = []
n = 2
#n = int(input("Enter no. of Humans: "))

with open("file.txt") as fin:
	#global n
	lines = fin.readlines()
	n = int(lines[0])
	lines = lines[1:]
	for line in lines:
		a,b = map(int,line.split())
		mu.append(a)
		var.append(b)
		
def getGaussianRandom(mu, var):
	x = random.gauss(mu,var)
	#x = abs(x)
	#x = min(x,1.1)
	#x = max(x,0.5)	
	return float("{0:.0f}".format(x))

for i in range(n):
	human_db[str(i)] = {
	"pt":getGaussianRandom(mu[i], var[i]),
	"ft":-1,
	"status":True,
	"jobs":0, 
	"mean":mu[i],
	"var":var[i],
	"start_timelist":[],
	"end_timelist":[]
	}
	r_request[i] = 1

def serve_req(h_id, curr_time):
	human_db[h_id]["status"] = False
	pt = getGaussianRandom(human_db[h_id]["mean"], human_db[h_id]["var"])
	human_db[h_id]["ft"] = curr_time + pt #human_db[h_id]["pt"]
	#print(h_id,"start",curr_time,"Fin_time",human_db[h_id]["ft"],"Process Time: ",pt )
	human_db[h_id]["start_timelist"].append(curr_time)

def call_r(curr_time):
	global human_db, r_free_status, r_curr_req_serving, r_request, r_finished_time
	if(not r_free_status):				#Robot is Working then see if finish time has reached
		if(curr_time == r_finished_time):	#If finished time has reached
			for i in range(n):
				if(i == r_curr_req_serving):
					serve_req(str(i), curr_time)
			r_curr_req_serving = -1 
			r_free_status = True
		else: 
			return

	h_flag = True
	for i in range(n):
		if(human_db[str(i)]["status"]==True):
			h_flag = False
			break;
	if(h_flag):
		return
	
	req_h_id = min(r_request, key=r_request.get)	
	r_request[req_h_id] = INT_MAX 		
	r_finished_time = curr_time + r_pt
	r_curr_req_serving = req_h_id

	r_free_status = False

def call_h(h_id, curr_time):
	global human_db, r_request
	if(human_db[h_id]["status"]): 
		return
	if(curr_time == human_db[h_id]["ft"]):
		human_db[h_id]["status"] = True
		r_request[int(h_id)] = curr_time
		#######r_request["pt"] =getGaussianRandom(human_db[h_id]["mean"], human_db[h_id]["var"])	
		human_db[h_id]["jobs"]+=1	
		human_db[h_id]["end_timelist"].append(curr_time)
times = []
jobs_list = []
working_time_list = []
for x in range(n):
	jobs_list.append([])
	working_time_list.append([])

for time in range(1,MAX_TIME+1):
	call_r(time)
	for i in range(n):
		call_h(str(i), time)
	times.append(time)

for i in range(n):
	job = 0
	for val_id in range(min(len(human_db[str(i)]["start_timelist"]),len(human_db[str(i)]["end_timelist"]))):
		working_time_list[i].append(human_db[str(i)]["start_timelist"][val_id])
		working_time_list[i].append(human_db[str(i)]["end_timelist"][val_id])
		jobs_list[i].append(job)
		jobs_list[i].append(job+1)
		job+=1
total_jobs = str(sum([jobs_list[x][-1] for x in range(n)]))

plt.xlabel('Time')
plt.ylabel('Number of Jobs')
plt.title('Total Jobs for FCFS:' + total_jobs)
legends = []
for i in range(n):
	t=plt.plot(working_time_list[i], jobs_list[i], label="Mean:" + str(human_db[str(i)]["mean"]) + ", Var:" + str(human_db[str(i)]["var"]) + ", Jobs:" + str(human_db[str(i)]["jobs"]) )
	legends.append(t[0])
#plt.plot(times, jobs2, 'g--', label='Human2')
plt.legend(handles=legends)
#plt.savefig("FCFS/"+sys.argv[1])
plt.show()

print("FCFS:", total_jobs)

