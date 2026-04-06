from __future__ import absolute_import, division, print_function
import utilities
import cluster
import workload


import matplotlib.pyplot as plt
import csv
import numpy as np
from itertools import chain
import seaborn as sns
import pandas as pd
# Bokeh imports
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.transform import transform
from bokeh.layouts import gridplot
import constants
from rm_environment import ClusterEnv
import os
from datetime import datetime
import copy
import heapq
from queue import PriorityQueue
import random

# Function to save lists to CSV files
def save_to_csv(filename, header, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Base Scheduler Class
class BaseScheduler:
    def __init__(self, name):
        self.name = name
        self.reset_metrics()
    
    def reset_metrics(self):
        self.total_cost = 0
        self.total_time = 0
        self.completed_jobs = 0
        self.good_placement = 0
        self.cpu_utilization_history = []
        self.mem_utilization_history = []
        self.throughput_history = []
        self.adherence_history = []
        self.episode_costs = []
        self.episode_times = []
        self.rewards = []
        self.start_time = 0
        self.end_time = 0
    
    def schedule(self, jobs, vms):
        raise NotImplementedError("Subclass must implement schedule method")
    
    def calculate_metrics(self, jobs, vms):
        # Calculate cost
        cost = sum(vm.price * vm.used_time for vm in vms)
        
        # Calculate average time
        total_duration = sum(job.duration for job in jobs if job.finished)
        avg_time = total_duration / len(jobs) if jobs else 0
        
        # Calculate resource utilization
        total_cpu = sum(vm.cpu for vm in vms)
        total_mem = sum(vm.mem for vm in vms)
        
        cpu_used = sum(vm.cpu - vm.cpu_now for vm in vms)
        mem_used = sum(vm.mem - vm.mem_now for vm in vms)
        
        cpu_utilization = cpu_used / total_cpu if total_cpu > 0 else 0
        mem_utilization = mem_used / total_mem if total_mem > 0 else 0
        avg_utilization = (cpu_utilization + mem_utilization) / 2
        
        # Calculate throughput
        total_time = self.end_time - self.start_time if self.end_time > self.start_time else 1
        throughput = self.completed_jobs / total_time * 1000  # Scale for visibility
        
        # Calculate deadline adherence
        on_time_jobs = sum(1 for job in jobs if job.finished and job.finish_time <= job.deadline)
        adherence = on_time_jobs / len(jobs) if jobs else 0
        
        return cost, avg_time, avg_utilization, throughput, adherence

# FIFO Scheduler
class FIFOScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("FIFO")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Sort jobs by arrival time
        jobs.sort(key=lambda x: x.arrival_time)
        
        current_time = 0
        self.start_time = jobs[0].arrival_time if jobs else 0
        job_queue = []
        
        for job in jobs:
            current_time = max(current_time, job.arrival_time)
            
            # Find earliest available VM with sufficient resources
            best_vm = None
            earliest_time = float('inf')
            
            for vm in vms:
                if vm.cpu >= job.cpu and vm.mem >= job.mem:
                    available_time = max(current_time, vm.stop_use_clock)
                    if available_time < earliest_time:
                        earliest_time = available_time
                        best_vm = vm
            
            if best_vm:
                job.start_time = earliest_time
                job.finish_time = job.start_time + job.duration
                job.running = True
                job.finished = True
                
                # Update VM
                best_vm.cpu_now = best_vm.cpu - job.cpu
                best_vm.mem_now = best_vm.mem - job.mem
                best_vm.used_time += job.duration
                best_vm.stop_use_clock = job.finish_time
                
                self.completed_jobs += 1
                self.good_placement += 1
                
                # Track metrics
                self.cpu_utilization_history.append(job.cpu / sum(vm.cpu for vm in vms))
                self.mem_utilization_history.append(job.mem / sum(vm.mem for vm in vms))
                self.rewards.append(10)  # Positive reward for successful placement
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        # Calculate final metrics
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# FCFS Scheduler (First Come First Served - similar to FIFO but with different implementation)
class FCFSScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("FCFS")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Sort by arrival time
        jobs.sort(key=lambda x: x.arrival_time)
        
        current_time = 0
        self.start_time = jobs[0].arrival_time if jobs else 0
        
        for job in jobs:
            current_time = max(current_time, job.arrival_time)
            
            # Find first available VM
            for vm in vms:
                if vm.cpu >= job.cpu and vm.mem >= job.mem:
                    available_time = max(current_time, vm.stop_use_clock)
                    job.start_time = available_time
                    job.finish_time = job.start_time + job.duration
                    job.running = True
                    job.finished = True
                    
                    vm.cpu_now = vm.cpu - job.cpu
                    vm.mem_now = vm.mem - job.mem
                    vm.used_time += job.duration
                    vm.stop_use_clock = job.finish_time
                    
                    self.completed_jobs += 1
                    self.good_placement += 1
                    
                    self.cpu_utilization_history.append(job.cpu / sum(vm.cpu for vm in vms))
                    self.mem_utilization_history.append(job.mem / sum(vm.mem for vm in vms))
                    self.rewards.append(10)
                    break
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# Fair Scheduler
class FairScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("Fair")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Group jobs by type for fair allocation
        job_types = {}
        for job in jobs:
            if job.type not in job_types:
                job_types[job.type] = []
            job_types[job.type].append(job)
        
        # Sort each type by arrival time
        for job_type in job_types:
            job_types[job_type].sort(key=lambda x: x.arrival_time)
        
        self.start_time = min(job.arrival_time for job in jobs) if jobs else 0
        current_time = self.start_time
        
        # Round-robin through job types
        type_keys = list(job_types.keys())
        type_index = 0
        
        while any(job_types.values()):
            current_type = type_keys[type_index % len(type_keys)]
            
            if job_types[current_type]:
                job = job_types[current_type].pop(0)
                current_time = max(current_time, job.arrival_time)
                
                # Find best fit VM
                best_vm = None
                min_waste = float('inf')
                
                for vm in vms:
                    if vm.cpu >= job.cpu and vm.mem >= job.mem:
                        waste = (vm.cpu - job.cpu) + (vm.mem - job.mem)
                        if waste < min_waste:
                            min_waste = waste
                            best_vm = vm
                
                if best_vm:
                    available_time = max(current_time, best_vm.stop_use_clock)
                    job.start_time = available_time
                    job.finish_time = job.start_time + job.duration
                    job.running = True
                    job.finished = True
                    
                    best_vm.cpu_now = best_vm.cpu - job.cpu
                    best_vm.mem_now = best_vm.mem - job.mem
                    best_vm.used_time += job.duration
                    best_vm.stop_use_clock = job.finish_time
                    
                    self.completed_jobs += 1
                    self.good_placement += 1
                    
                    self.cpu_utilization_history.append(job.cpu / sum(vm.cpu for vm in vms))
                    self.mem_utilization_history.append(job.mem / sum(vm.mem for vm in vms))
                    self.rewards.append(10)
            
            type_index += 1
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# Capacity Scheduler
class CapacityScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("Capacity")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Assign capacity quotas to job types
        job_types = set(job.type for job in jobs)
        quota_per_type = 1.0 / len(job_types) if job_types else 1.0
        
        # Group jobs by type
        job_queues = {}
        for job_type in job_types:
            job_queues[job_type] = [job for job in jobs if job.type == job_type]
            job_queues[job_type].sort(key=lambda x: x.arrival_time)
        
        self.start_time = min(job.arrival_time for job in jobs) if jobs else 0
        current_time = self.start_time
        
        # Calculate resource allocation per type
        total_cpu = sum(vm.cpu for vm in vms)
        total_mem = sum(vm.mem for vm in vms)
        
        while any(job_queues.values()):
            for job_type in job_types:
                if job_queues[job_type]:
                    job = job_queues[job_type].pop(0)
                    current_time = max(current_time, job.arrival_time)
                    
                    # Find VM with capacity within quota
                    best_vm = None
                    for vm in vms:
                        if vm.cpu >= job.cpu and vm.mem >= job.mem:
                            # Check if this allocation is within capacity quota
                            current_usage = (vm.cpu - vm.cpu_now) / total_cpu
                            if current_usage < quota_per_type:
                                best_vm = vm
                                break
                    
                    # If no VM within quota, use any available VM
                    if not best_vm:
                        for vm in vms:
                            if vm.cpu >= job.cpu and vm.mem >= job.mem:
                                best_vm = vm
                                break
                    
                    if best_vm:
                        available_time = max(current_time, best_vm.stop_use_clock)
                        job.start_time = available_time
                        job.finish_time = job.start_time + job.duration
                        job.running = True
                        job.finished = True
                        
                        best_vm.cpu_now = best_vm.cpu - job.cpu
                        best_vm.mem_now = best_vm.mem - job.mem
                        best_vm.used_time += job.duration
                        best_vm.stop_use_clock = job.finish_time
                        
                        self.completed_jobs += 1
                        self.good_placement += 1
                        
                        self.cpu_utilization_history.append(job.cpu / total_cpu)
                        self.mem_utilization_history.append(job.mem / total_mem)
                        self.rewards.append(10)
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# Round Robin Scheduler
class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("RoundRobin")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Sort by arrival time
        jobs.sort(key=lambda x: x.arrival_time)
        
        self.start_time = jobs[0].arrival_time if jobs else 0
        current_time = self.start_time
        vm_index = 0
        
        for job in jobs:
            current_time = max(current_time, job.arrival_time)
            
            # Try VMs in round-robin fashion
            attempts = 0
            while attempts < len(vms):
                vm = vms[vm_index]
                
                if vm.cpu >= job.cpu and vm.mem >= job.mem:
                    available_time = max(current_time, vm.stop_use_clock)
                    job.start_time = available_time
                    job.finish_time = job.start_time + job.duration
                    job.running = True
                    job.finished = True
                    
                    vm.cpu_now = vm.cpu - job.cpu
                    vm.mem_now = vm.mem - job.mem
                    vm.used_time += job.duration
                    vm.stop_use_clock = job.finish_time
                    
                    self.completed_jobs += 1
                    self.good_placement += 1
                    
                    self.cpu_utilization_history.append(job.cpu / sum(vm.cpu for vm in vms))
                    self.mem_utilization_history.append(job.mem / sum(vm.mem for vm in vms))
                    self.rewards.append(10)
                    break
                
                vm_index = (vm_index + 1) % len(vms)
                attempts += 1
            
            vm_index = (vm_index + 1) % len(vms)
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# Min-Min Scheduler
class MinMinScheduler(BaseScheduler):
    def __init__(self):
        super().__init__("MinMin")
    
    def schedule(self, jobs, vms):
        self.reset_metrics()
        jobs = copy.deepcopy(jobs)
        vms = copy.deepcopy(vms)
        
        # Sort by arrival time
        jobs.sort(key=lambda x: x.arrival_time)
        
        self.start_time = jobs[0].arrival_time if jobs else 0
        current_time = self.start_time
        unscheduled_jobs = list(jobs)
        
        while unscheduled_jobs:
            # Find job with minimum completion time
            min_completion_time = float('inf')
            best_job = None
            best_vm = None
            
            for job in unscheduled_jobs:
                if job.arrival_time <= current_time:
                    for vm in vms:
                        if vm.cpu >= job.cpu and vm.mem >= job.mem:
                            available_time = max(current_time, vm.stop_use_clock)
                            completion_time = available_time + job.duration
                            
                            if completion_time < min_completion_time:
                                min_completion_time = completion_time
                                best_job = job
                                best_vm = vm
            
            if best_job and best_vm:
                current_time = max(current_time, best_job.arrival_time)
                available_time = max(current_time, best_vm.stop_use_clock)
                
                best_job.start_time = available_time
                best_job.finish_time = best_job.start_time + best_job.duration
                best_job.running = True
                best_job.finished = True
                
                best_vm.cpu_now = best_vm.cpu - best_job.cpu
                best_vm.mem_now = best_vm.mem - best_job.mem
                best_vm.used_time += best_job.duration
                best_vm.stop_use_clock = best_job.finish_time
                
                self.completed_jobs += 1
                self.good_placement += 1
                
                self.cpu_utilization_history.append(best_job.cpu / sum(vm.cpu for vm in vms))
                self.mem_utilization_history.append(best_job.mem / sum(vm.mem for vm in vms))
                self.rewards.append(10)
                
                unscheduled_jobs.remove(best_job)
                current_time = best_job.finish_time
            else:
                # No suitable job found, advance time
                if unscheduled_jobs:
                    current_time = min(job.arrival_time for job in unscheduled_jobs if job.arrival_time > current_time)
        
        self.end_time = max(job.finish_time for job in jobs if hasattr(job, 'finish_time')) if jobs else 0
        
        cost, avg_time, utilization, throughput, adherence = self.calculate_metrics(jobs, vms)
        self.episode_costs.append(cost)
        self.episode_times.append(avg_time)
        self.throughput_history.append(throughput)
        self.adherence_history.append(adherence)
        
        return jobs, vms

# Plotting functions (same as C51)
def plot_throughput_heatmap(throughput_data, scheduler_names, downsample_factor=1, grid_size=(10, 10)):
    # Create heatmap for multiple schedulers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (scheduler_name, throughput_list) in enumerate(zip(scheduler_names, throughput_data)):
        if i < len(axes):
            # Downsample and reshape
            throughput_list = throughput_list[::downsample_factor]
            if len(throughput_list) >= grid_size[0] * grid_size[1]:
                throughput_matrix = np.array(throughput_list[:grid_size[0]*grid_size[1]]).reshape(grid_size)
            else:
                # Pad with zeros if not enough data
                padded_list = throughput_list + [0] * (grid_size[0]*grid_size[1] - len(throughput_list))
                throughput_matrix = np.array(padded_list[:grid_size[0]*grid_size[1]]).reshape(grid_size)
            
            sns.heatmap(throughput_matrix, cmap='viridis', annot=True, fmt=".1f", ax=axes[i])
            axes[i].set_title(f'{scheduler_name} Throughput Heatmap')
            axes[i].set_ylabel('Job Throughput %')
            axes[i].set_xlabel('Episode')
    
    plt.tight_layout()
    plt.show()

def plot_scheduler_comparison_bokeh(schedulers_data, metric_name, y_label, title):
    p = figure(title=title, x_axis_label='Scheduler', y_axis_label=y_label, width=800, height=400)
    
    scheduler_names = list(schedulers_data.keys())
    values = [schedulers_data[name] for name in scheduler_names]
    
    p.vbar(x=scheduler_names, top=values, width=0.8, alpha=0.7)
    p.xgrid.grid_line_color = None
    p.grid.visible = True
    
    output_file(f"{metric_name}_comparison.html")
    show(p)

def plot_comparison_metrics(results):
    # Create comparison plots
    schedulers = list(results.keys())
    
    # Cost comparison
    costs = [results[scheduler]['cost'] for scheduler in schedulers]
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.bar(schedulers, costs, alpha=0.7)
    plt.title('Cost Comparison')
    plt.ylabel('Cost')
    plt.xticks(rotation=45)
    
    # Time comparison
    times = [results[scheduler]['avg_time'] for scheduler in schedulers]
    plt.subplot(2, 3, 2)
    plt.bar(schedulers, times, alpha=0.7)
    plt.title('Average Time Comparison')
    plt.ylabel('Average Time')
    plt.xticks(rotation=45)
    
    # Utilization comparison
    utilizations = [results[scheduler]['utilization'] for scheduler in schedulers]
    plt.subplot(2, 3, 3)
    plt.bar(schedulers, utilizations, alpha=0.7)
    plt.title('Resource Utilization Comparison')
    plt.ylabel('Utilization')
    plt.xticks(rotation=45)
    
    # Throughput comparison
    throughputs = [results[scheduler]['throughput'] for scheduler in schedulers]
    plt.subplot(2, 3, 4)
    plt.bar(schedulers, throughputs, alpha=0.7)
    plt.title('Throughput Comparison')
    plt.ylabel('Throughput')
    plt.xticks(rotation=45)
    
    # Adherence comparison
    adherences = [results[scheduler]['adherence'] for scheduler in schedulers]
    plt.subplot(2, 3, 5)
    plt.bar(schedulers, adherences, alpha=0.7)
    plt.title('Deadline Adherence Comparison')
    plt.ylabel('Adherence')
    plt.xticks(rotation=45)
    
    # Good placement comparison
    good_placements = [results[scheduler]['good_placement'] for scheduler in schedulers]
    plt.subplot(2, 3, 6)
    plt.bar(schedulers, good_placements, alpha=0.7)
    plt.title('Good Placement Comparison')
    plt.ylabel('Good Placements')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def run_benchmark_comparison():
    # Import required modules
    import cluster
    import workload
    import utilities
    
    # Initialize the system properly
    utilities.load_config()
    workload.read_workload()
    cluster.init_cluster()  # This is crucial - must be called before creating environment
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_benchmark")
    output_dir = os.path.join(constants.root, 'output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize schedulers
    schedulers = [
        FIFOScheduler(),
        FCFSScheduler(),
        FairScheduler(),
        CapacityScheduler(),
        RoundRobinScheduler(),
        MinMinScheduler()
    ]
    
    # NOW initialize environment after cluster is properly set up
    env = ClusterEnv()
    
    # Rest of the code remains the same...

    
    # Create CSV file for results
    results_file = os.path.join(output_dir, 'benchmark_results.csv')
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scheduler", "Cost", "AvgTime", "Utilization", "Throughput", "Adherence", "GoodPlacement"])
    
    results = {}
    all_throughput_data = []
    scheduler_names = []
    
    # Run each scheduler
    for scheduler in schedulers:
        print(f"Running {scheduler.name} scheduler...")
        
        # Get fresh jobs and VMs
        jobs = copy.deepcopy(env.jobs)
        vms = copy.deepcopy(env.vms)
        
        # Run scheduler
        scheduled_jobs, scheduled_vms = scheduler.schedule(jobs, vms)
        
        # Calculate metrics
        cost, avg_time, utilization, throughput, adherence = scheduler.calculate_metrics(scheduled_jobs, scheduled_vms)
        
        # Store results
        results[scheduler.name] = {
            'cost': cost,
            'avg_time': avg_time,
            'utilization': utilization,
            'throughput': throughput,
            'adherence': adherence,
            'good_placement': scheduler.good_placement,
            'rewards': scheduler.rewards,
            'cpu_utilization': scheduler.cpu_utilization_history,
            'mem_utilization': scheduler.mem_utilization_history,
            'episode_costs': scheduler.episode_costs,
            'episode_times': scheduler.episode_times
        }
        
        # Collect data for plotting
        all_throughput_data.append(scheduler.throughput_history)
        scheduler_names.append(scheduler.name)
        
        # Write to CSV
        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([scheduler.name, cost, avg_time, utilization, throughput, adherence, scheduler.good_placement])
        
        print(f"{scheduler.name} - Cost: {cost:.2f}, Avg Time: {avg_time:.2f}, Utilization: {utilization:.2f}, Throughput: {throughput:.2f}, Adherence: {adherence:.2f}")
    
    # Generate plots
    plot_comparison_metrics(results)
    
    # Generate Bokeh plots
    plot_scheduler_comparison_bokeh(
        {name: results[name]['cost'] for name in scheduler_names},
        'cost', 'Cost', 'Cost Comparison Across Schedulers'
    )
    
    plot_scheduler_comparison_bokeh(
        {name: results[name]['throughput'] for name in scheduler_names},
        'throughput', 'Throughput', 'Throughput Comparison Across Schedulers'
    )
    
    plot_scheduler_comparison_bokeh(
        {name: results[name]['adherence'] for name in scheduler_names},
        'adherence', 'Adherence', 'Deadline Adherence Comparison Across Schedulers'
    )
    
    # Save individual scheduler data
    for scheduler_name in scheduler_names:
        scheduler_dir = os.path.join(output_dir, scheduler_name.lower())
        os.makedirs(scheduler_dir, exist_ok=True)
        
        # Save detailed metrics
        save_to_csv(
            os.path.join(scheduler_dir, 'rewards.csv'),
            ['Rewards'],
            [[val] for val in results[scheduler_name]['rewards']]
        )
        
        save_to_csv(
            os.path.join(scheduler_dir, 'utilization.csv'),
            ['CPU Utilization', 'Memory Utilization'],
            list(zip(results[scheduler_name]['cpu_utilization'], results[scheduler_name]['mem_utilization']))
        )
        
        save_to_csv(
            os.path.join(scheduler_dir, 'episode_costs.csv'),
            ['Episode Costs'],
            [[val] for val in results[scheduler_name]['episode_costs']]
        )
        
        save_to_csv(
            os.path.join(scheduler_dir, 'episode_times.csv'),
            ['Episode Times'],
            [[val] for val in results[scheduler_name]['episode_times']]
        )
    
    # Create summary comparison table
    summary_file = os.path.join(output_dir, 'summary_comparison.csv')
    with open(summary_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric'] + scheduler_names)
        
        metrics = ['Cost', 'Avg Time', 'Utilization', 'Throughput', 'Adherence', 'Good Placement']
        metric_keys = ['cost', 'avg_time', 'utilization', 'throughput', 'adherence', 'good_placement']
        
        for metric, key in zip(metrics, metric_keys):
            row = [metric] + [results[name][key] for name in scheduler_names]
            writer.writerow(row)
    
    print(f"\nBenchmark comparison complete! Results saved to: {output_dir}")
    print(f"Summary comparison saved to: {summary_file}")
    
    return results

# Main execution function
if __name__ == "__main__":
    print("Starting Benchmark Scheduler Comparison...")
    results = run_benchmark_comparison()
    print("Benchmark comparison completed successfully!")
