import pandas as pd

# Load the CSV file
file_path = 'D:\sumit\RM_DeepRL-master\input\jobs_burst.csv'
jobs_df = pd.read_csv(file_path, header=None)
jobs_df.columns = ['Job_ID', 'Arrival_Time', 'Duration', 'Other1', 'Other2', 'Other3', 'Other4']

# Extract only the necessary columns: Job_ID, Arrival_Time, Duration
jobs_df = jobs_df[['Job_ID', 'Arrival_Time', 'Duration']]

# Convert columns to appropriate data types
jobs_df['Arrival_Time'] = jobs_df['Arrival_Time'].astype(int)
jobs_df['Duration'] = jobs_df['Duration'].astype(int)

def fcfs(jobs):
    jobs = jobs.sort_values(by='Arrival_Time').reset_index(drop=True)
    current_time = 0
    total_duration = 0
    
    for i, job in jobs.iterrows():
        if current_time < job['Arrival_Time']:
            current_time = job['Arrival_Time']
        current_time += job['Duration']
        total_duration += (current_time - job['Arrival_Time'])
    
    average_duration = total_duration / len(jobs)
    return average_duration

def sjf(jobs):
    jobs = jobs.sort_values(by=['Arrival_Time', 'Duration']).reset_index(drop=True)
    current_time = 0
    total_duration = 0
    completed_jobs = 0
    remaining_jobs = jobs.copy()
    
    while not remaining_jobs.empty:
        available_jobs = remaining_jobs[remaining_jobs['Arrival_Time'] <= current_time]
        
        if available_jobs.empty:
            current_time = remaining_jobs['Arrival_Time'].min()
            available_jobs = remaining_jobs[remaining_jobs['Arrival_Time'] <= current_time]
        
        next_job = available_jobs.sort_values(by='Duration').iloc[0]
        remaining_jobs = remaining_jobs[remaining_jobs['Job_ID'] != next_job['Job_ID']]
        
        current_time += next_job['Duration']
        total_duration += (current_time - next_job['Arrival_Time'])
        completed_jobs += 1
    
    average_duration = total_duration / len(jobs)
    return average_duration

def round_robin(jobs, quantum):
    jobs = jobs.sort_values(by='Arrival_Time').reset_index(drop=True)
    current_time = 0
    total_duration = 0
    queue = jobs.copy()
    job_indices = list(range(len(jobs)))
    job_indices_queue = []
    
    while queue.shape[0] > 0 or len(job_indices_queue) > 0:
        while len(job_indices) > 0 and jobs.loc[job_indices[0], 'Arrival_Time'] <= current_time:
            job_indices_queue.append(job_indices.pop(0))
        
        if len(job_indices_queue) == 0:
            current_time = jobs.loc[job_indices[0], 'Arrival_Time']
            continue
        
        current_job_index = job_indices_queue.pop(0)
        job = jobs.loc[current_job_index]
        
        if job['Duration'] <= quantum:
            current_time += job['Duration']
            total_duration += (current_time - job['Arrival_Time'])
            queue = queue[queue['Job_ID'] != job['Job_ID']]
        else:
            current_time += quantum
            jobs.at[current_job_index, 'Duration'] -= quantum
            while len(job_indices) > 0 and jobs.loc[job_indices[0], 'Arrival_Time'] <= current_time:
                job_indices_queue.append(job_indices.pop(0))
            job_indices_queue.append(current_job_index)
    
    average_duration = total_duration / len(jobs)
    return average_duration

# Compare the algorithms
fcfs_avg_duration = fcfs(jobs_df)
sjf_avg_duration = sjf(jobs_df)
rr_avg_duration = round_robin(jobs_df, quantum=4)

print("FCFS Average Job Duration:", fcfs_avg_duration)
print("SJF Average Job Duration:", sjf_avg_duration)
print("Round Robin (Quantum=4) Average Job Duration:", rr_avg_duration)
