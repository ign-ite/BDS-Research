from __future__ import absolute_import, division, print_function

import os
import sys
import json
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from itertools import chain

# Bokeh imports
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import HoverTool

# Import your existing modules
import constants
import utilities
import cluster
import workload
from rm_environment import ClusterEnv

# Import ALL the algorithms you want to compare
import REINFORCE_tfagent
import DQN_tfagent
import C51_tfagent
import R_DQN_tfagent

# Import baseline scheduler
from baseline_schedulers import (
    FIFOScheduler, FCFSScheduler, FairScheduler, 
    CapacityScheduler, RoundRobinScheduler, MinMinScheduler,
    save_to_csv
)

import copy

class ComprehensiveComparisonManager:
    def __init__(self):
        self.results = {}
        self.output_dir = None
        # Categorize algorithms for better visualization
        self.basic_rl = ['REINFORCE', 'DQN']
        self.advanced_rl = ['Rainbow_DQN', 'C51']
        self.baseline_schedulers = ['FIFO', 'FCFS', 'Fair', 'Capacity', 'RoundRobin', 'MinMin']
        self.all_methods = self.basic_rl + self.advanced_rl + self.baseline_schedulers
        
    def setup_output_directory(self):
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_complete_comparison")
        self.output_dir = os.path.join(constants.root, 'output', timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        for method in self.all_methods:
            os.makedirs(os.path.join(self.output_dir, method.lower()), exist_ok=True)
            
        print(f"Results will be saved to: {self.output_dir}")
        
    def initialize_environment(self):
        """Initialize the cluster environment"""
        utilities.load_config()
        workload.read_workload()
        cluster.init_cluster()
        
    def run_rl_algorithm(self, algorithm_name, num_iterations=10000):
        """Run specific RL algorithm with optimized parameters"""
        print(f"\n{'='*60}")
        print(f"Training {algorithm_name} Algorithm")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if algorithm_name == 'REINFORCE':
                REINFORCE_tfagent.train_reinforce(
                    num_iterations=num_iterations,
                    collect_episodes_per_iteration=6,
                    replay_buffer_max_length=10000,
                    fc_layer_params=(100,),
                    learning_rate=9e-4,
                    log_interval=200,
                    num_eval_episodes=10,
                    eval_interval=1000
                )
                
            elif algorithm_name == 'DQN':
                DQN_tfagent.train_dqn(
                    num_iterations=num_iterations,
                    initial_collect_steps=1000,
                    collect_steps_per_iteration=10,
                    replay_buffer_max_length=100000,
                    fc_layer_params=(200,),
                    batch_size=128,
                    learning_rate=1e-3,
                    log_interval=200,
                    num_eval_episodes=10,
                    eval_interval=1000
                )
                
            elif algorithm_name == 'C51':
                C51_tfagent.train_c51_dqn(
                    num_iterations=num_iterations,
                    initial_collect_steps=1000,
                    collect_steps_per_iteration=10,
                    replay_buffer_max_length=100000,
                    fc_layer_params=(200,100),
                    batch_size=128,
                    learning_rate=1e-3,
                    log_interval=200,
                    num_eval_episodes=10,
                    eval_interval=1000
                )
                
            elif algorithm_name == 'Rainbow_DQN':
                R_DQN_tfagent.train_rainbow_dqn(
                    num_iterations=num_iterations,
                    initial_collect_steps=1000,
                    collect_steps_per_iteration=10,
                    replay_buffer_max_length=100000,
                    fc_layer_params=(200,),
                    batch_size=128,
                    initial_learning_rate=9e-4,
                    log_interval=200,
                    num_eval_episodes=10,
                    eval_interval=1000
                )
                
        except Exception as e:
            print(f"Error training {algorithm_name}: {str(e)}")
            return None
            
        end_time = time.time()
        training_time = end_time - start_time
        
        # Load results
        results = self.load_rl_results(algorithm_name)
        results['training_time'] = training_time
        
        print(f"{algorithm_name} training completed in {training_time:.2f} seconds")
        return results
        
    def load_rl_results(self, algorithm_name):
        """Load results from RL algorithm output files"""
        results = {
            'cost': 0, 'avg_time': 0, 'utilization': 0, 'throughput': 0,
            'adherence': 0, 'good_placement': 0, 'rewards': [],
            'episode_costs': [], 'episode_times': [], 'cpu_utilization': [],
            'mem_utilization': [], 'throughput_history': [], 'adherence_history': []
        }
        
        # Find the most recent output directory
        output_dirs = [d for d in os.listdir(os.path.join(constants.root, 'output')) 
                      if d.startswith('2') and os.path.isdir(os.path.join(constants.root, 'output', d))]
        
        if not output_dirs:
            print(f"No output directories found for {algorithm_name}")
            return results
            
        # Get the most recent directory
        latest_dir = max(output_dirs)
        results_path = os.path.join(constants.root, 'output', latest_dir)
        
        try:
            # Load episode costs
            costs_file = os.path.join(results_path, 'episode_costs.csv')
            if os.path.exists(costs_file):
                costs_df = pd.read_csv(costs_file)
                costs = costs_df['Episode Costs'].tolist()
                results['episode_costs'] = costs
                results['cost'] = np.mean(costs) if costs else 0
                
            # Load episode times
            times_file = os.path.join(results_path, 'episode_time.csv')
            if os.path.exists(times_file):
                times_df = pd.read_csv(times_file)
                times = times_df['Episode Time'].tolist()
                results['episode_times'] = times
                results['avg_time'] = np.mean(times) if times else 0
                
            # Load utilization
            util_file = os.path.join(results_path, 'utilization.csv')
            if os.path.exists(util_file):
                util_df = pd.read_csv(util_file)
                cpu_util = util_df['CPU Utilization'].tolist()
                mem_util = util_df['Memory Utilization'].tolist()
                results['cpu_utilization'] = cpu_util
                results['mem_utilization'] = mem_util
                results['utilization'] = (np.mean(cpu_util) + np.mean(mem_util)) / 2
                
            # Load throughput
            throughput_file = os.path.join(results_path, 'throughput.csv')
            if os.path.exists(throughput_file):
                throughput_df = pd.read_csv(throughput_file)
                throughput_data = throughput_df['Throughput'].tolist()
                results['throughput_history'] = throughput_data
                results['throughput'] = np.mean(throughput_data) if throughput_data else 0
                
            # Load adherence
            adherence_file = os.path.join(results_path, 'adherence.csv')
            if os.path.exists(adherence_file):
                adherence_df = pd.read_csv(adherence_file)
                adherence_data = adherence_df['Adherence'].tolist()
                results['adherence_history'] = adherence_data
                results['adherence'] = np.mean(adherence_data) if adherence_data else 0
                
            # Load rewards
            rewards_file = os.path.join(results_path, 'rewards.csv')
            if os.path.exists(rewards_file):
                rewards_df = pd.read_csv(rewards_file)
                rewards_data = rewards_df['Rewards'].tolist()
                results['rewards'] = rewards_data
                results['good_placement'] = sum(1 for r in rewards_data if r > 0)
                
            print(f"Successfully loaded results for {algorithm_name}")
            print(f"  Cost: {results['cost']:.2f}")
            print(f"  Avg Time: {results['avg_time']:.2f}")
            print(f"  Utilization: {results['utilization']:.2f}")
            print(f"  Throughput: {results['throughput']:.2f}")
            print(f"  Adherence: {results['adherence']:.2f}")
                
        except Exception as e:
            print(f"Error loading results for {algorithm_name}: {str(e)}")
            
        return results
        
    def run_baseline_schedulers(self):
        """Run baseline schedulers"""
        print(f"\n{'='*60}")
        print("Running Baseline Schedulers")
        print(f"{'='*60}")
        
        schedulers = [
            FIFOScheduler(),
            FCFSScheduler(),
            FairScheduler(),
            CapacityScheduler(),
            RoundRobinScheduler(),
            MinMinScheduler()
        ]
        
        env = ClusterEnv()
        baseline_results = {}
        
        for scheduler in schedulers:
            print(f"Running {scheduler.name} scheduler...")
            start_time = time.time()
            
            jobs = copy.deepcopy(env.jobs)
            vms = copy.deepcopy(env.vms)
            
            scheduled_jobs, scheduled_vms = scheduler.schedule(jobs, vms)
            cost, avg_time, utilization, throughput, adherence = scheduler.calculate_metrics(scheduled_jobs, scheduled_vms)
            
            end_time = time.time()
            
            baseline_results[scheduler.name] = {
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
                'episode_times': scheduler.episode_times,
                'training_time': end_time - start_time
            }
            
            print(f"  {scheduler.name} - Cost: {cost:.2f}, Time: {avg_time:.2f}, "
                  f"Utilization: {utilization:.2f}, Throughput: {throughput:.2f}, Adherence: {adherence:.2f}")
                  
        return baseline_results
        
    def create_comprehensive_comparison_plots(self):
        """Create comprehensive comparison plots showcasing advanced RL superiority"""
        print(f"\n{'='*60}")
        print("Generating Comprehensive Comparison Plots")
        print(f"{'='*60}")
        
        methods = list(self.results.keys())
        
        # Extract metrics
        costs = [self.results[method]['cost'] for method in methods]
        times = [self.results[method]['avg_time'] for method in methods]
        utilizations = [self.results[method]['utilization'] for method in methods]
        throughputs = [self.results[method]['throughput'] for method in methods]
        adherences = [self.results[method]['adherence'] for method in methods]
        good_placements = [self.results[method]['good_placement'] for method in methods]
        training_times = [self.results[method].get('training_time', 0) for method in methods]
        
        # Create main comparison plot
        self.plot_layered_comparison(methods, costs, times, utilizations, throughputs, adherences, good_placements)
        
        # Create performance evolution plots
        self.plot_performance_evolution()
        
        # Create radar chart with categories
        self.plot_categorized_radar_chart(methods, costs, times, utilizations, throughputs, adherences)
        
        # Create performance ranking
        self.create_performance_ranking(methods, costs, times, utilizations, throughputs, adherences)
        
        # Create statistical significance analysis
        self.create_statistical_analysis()
        
    def plot_layered_comparison(self, methods, costs, times, utilizations, throughputs, adherences, good_placements):
        """Create layered comparison showing algorithm categories"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        
        # Define colors for different categories
        colors = []
        for method in methods:
            if method in self.advanced_rl:
                colors.append('#FF4444' if method == 'Rainbow_DQN' else '#CC0000')  # Red shades for advanced RL
            elif method in self.basic_rl:
                colors.append('#FFA500' if method == 'DQN' else '#FF8C00')  # Orange shades for basic RL
            else:
                colors.append('#87CEEB')  # Light blue for baseline schedulers
        
        # Cost comparison
        bars1 = axes[0, 0].bar(methods, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 0].set_title('Cost Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Cost')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Time comparison
        bars2 = axes[0, 1].bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 1].set_title('Average Job Completion Time\n(Lower is Better)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Average Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Utilization comparison
        bars3 = axes[0, 2].bar(methods, utilizations, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0, 2].set_title('Resource Utilization\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Utilization')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        bars4 = axes[1, 0].bar(methods, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[1, 0].set_title('System Throughput\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Throughput')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Adherence comparison
        bars5 = axes[1, 1].bar(methods, adherences, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[1, 1].set_title('Deadline Adherence\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Adherence')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Good placement comparison
        bars6 = axes[1, 2].bar(methods, good_placements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        axes[1, 2].set_title('Good Placement Count\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Good Placements')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for ax, bars, values in zip(axes.flatten(), [bars1, bars2, bars3, bars4, bars5, bars6], 
                                   [costs, times, utilizations, throughputs, adherences, good_placements]):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#FF4444', alpha=0.8, label='Advanced RL (Rainbow DQN)'),
            plt.Rectangle((0,0),1,1, facecolor='#CC0000', alpha=0.8, label='Advanced RL (C51)'),
            plt.Rectangle((0,0),1,1, facecolor='#FFA500', alpha=0.8, label='Basic RL (DQN)'),
            plt.Rectangle((0,0),1,1, facecolor='#FF8C00', alpha=0.8, label='Basic RL (REINFORCE)'),
            plt.Rectangle((0,0),1,1, facecolor='#87CEEB', alpha=0.8, label='Baseline Schedulers')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_layered_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_performance_evolution(self):
        """Plot performance evolution for RL algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rl_methods = [method for method in self.results.keys() if method in self.basic_rl + self.advanced_rl]
        
        # Define colors for consistency
        color_map = {
            'Rainbow_DQN': '#FF4444',
            'C51': '#CC0000',
            'DQN': '#FFA500',
            'REINFORCE': '#FF8C00'
        }
        
        # Rewards evolution
        axes[0, 0].set_title('Learning Curves: Rewards Evolution', fontsize=14, fontweight='bold')
        for method in rl_methods:
            if self.results[method]['rewards']:
                rewards = self.results[method]['rewards']
                if len(rewards) > 100:
                    smoothed_rewards = np.convolve(rewards, np.ones(100)/100, mode='valid')
                    axes[0, 0].plot(smoothed_rewards, label=method, linewidth=2.5, 
                                  color=color_map.get(method, 'gray'))
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Smoothed Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cost evolution
        axes[0, 1].set_title('Cost Evolution During Training', fontsize=14, fontweight='bold')
        for method in rl_methods:
            if self.results[method]['episode_costs']:
                costs = self.results[method]['episode_costs']
                axes[0, 1].plot(costs, label=method, linewidth=2.5, 
                              color=color_map.get(method, 'gray'))
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput evolution
        axes[1, 0].set_title('Throughput Evolution During Training', fontsize=14, fontweight='bold')
        for method in rl_methods:
            if self.results[method]['throughput_history']:
                throughput = self.results[method]['throughput_history']
                axes[1, 0].plot(throughput, label=method, linewidth=2.5, 
                              color=color_map.get(method, 'gray'))
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Throughput')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Utilization evolution
        axes[1, 1].set_title('CPU Utilization During Training', fontsize=14, fontweight='bold')
        for method in rl_methods:
            if self.results[method]['cpu_utilization']:
                cpu_util = self.results[method]['cpu_utilization']
                if isinstance(cpu_util[0], list):
                    cpu_util = list(chain.from_iterable(cpu_util))
                # Take only first 1000 points for clarity
                cpu_util = cpu_util[:1000]
                axes[1, 1].plot(cpu_util, label=method, linewidth=2.5, 
                              color=color_map.get(method, 'gray'))
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('CPU Utilization')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_categorized_radar_chart(self, methods, costs, times, utilizations, throughputs, adherences):
        """Create categorized radar chart"""
        metrics = ['Cost', 'Time', 'Utilization', 'Throughput', 'Adherence']
        
        # Prepare data
        data = {}
        for i, method in enumerate(methods):
            data[method] = [costs[i], times[i], utilizations[i], throughputs[i], adherences[i]]
        
        # Normalize data
        normalized_data = {}
        for i, metric in enumerate(metrics):
            values = [data[method][i] for method in methods]
            max_val = max(values) if values else 1
            min_val = min(values) if values else 0
            
            for method in methods:
                if method not in normalized_data:
                    normalized_data[method] = []
                
                if max_val != min_val:
                    if metric in ['Cost', 'Time']:
                        normalized_val = 1 - (data[method][i] - min_val) / (max_val - min_val)
                    else:
                        normalized_val = (data[method][i] - min_val) / (max_val - min_val)
                else:
                    normalized_val = 0.5
                
                normalized_data[method].append(normalized_val)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Plot advanced RL algorithms with thicker lines
        for method in methods:
            if method in self.advanced_rl:
                values = normalized_data[method]
                values = np.concatenate((values, [values[0]]))
                
                color = '#FF4444' if method == 'Rainbow_DQN' else '#CC0000'
                ax.plot(angles, values, 'o-', linewidth=4, label=method, color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
        
        # Plot basic RL algorithms with medium lines
        for method in methods:
            if method in self.basic_rl:
                values = normalized_data[method]
                values = np.concatenate((values, [values[0]]))
                
                color = '#FFA500' if method == 'DQN' else '#FF8C00'
                ax.plot(angles, values, 'o-', linewidth=3, label=method, color=color)
                ax.fill(angles, values, alpha=0.15, color=color)
        
        # Plot baseline schedulers with thinner lines
        for method in methods:
            if method in self.baseline_schedulers:
                values = normalized_data[method]
                values = np.concatenate((values, [values[0]]))
                
                ax.plot(angles, values, 'o-', linewidth=2, label=method, color='#87CEEB', alpha=0.7)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Performance Radar Chart\n(Advanced RL vs Basic RL vs Baseline Schedulers)', 
                    size=16, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'categorized_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_performance_ranking(self, methods, costs, times, utilizations, throughputs, adherences):
        """Create performance ranking table"""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE PERFORMANCE RANKING")
        print(f"{'='*100}")
        
        # Create ranking for each metric
        rankings = {}
        
        # Cost ranking (lower is better)
        cost_ranking = sorted(zip(methods, costs), key=lambda x: x[1])
        rankings['Cost'] = {method: rank+1 for rank, (method, _) in enumerate(cost_ranking)}
        
        # Time ranking (lower is better)
        time_ranking = sorted(zip(methods, times), key=lambda x: x[1])
        rankings['Time'] = {method: rank+1 for rank, (method, _) in enumerate(time_ranking)}
        
        # Utilization ranking (higher is better)
        util_ranking = sorted(zip(methods, utilizations), key=lambda x: x[1], reverse=True)
        rankings['Utilization'] = {method: rank+1 for rank, (method, _) in enumerate(util_ranking)}
        
        # Throughput ranking (higher is better)
        throughput_ranking = sorted(zip(methods, throughputs), key=lambda x: x[1], reverse=True)
        rankings['Throughput'] = {method: rank+1 for rank, (method, _) in enumerate(throughput_ranking)}
        
        # Adherence ranking (higher is better)
        adherence_ranking = sorted(zip(methods, adherences), key=lambda x: x[1], reverse=True)
        rankings['Adherence'] = {method: rank+1 for rank, (method, _) in enumerate(adherence_ranking)}
        
        # Calculate overall score (lower is better)
        overall_scores = {}
        for method in methods:
            score = sum(rankings[metric][method] for metric in rankings.keys())
            overall_scores[method] = score
        
        # Sort by overall score
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
        
        # Create ranking table
        print(f"{'Rank':<5} {'Method':<12} {'Cost':<6} {'Time':<6} {'Util':<6} {'Throughput':<11} {'Adherence':<10} {'Overall Score':<15}")
        print("-" * 100)
        
        for i, (method, score) in enumerate(overall_ranking):
            print(f"{i+1:<5} {method:<12} {rankings['Cost'][method]:<6} {rankings['Time'][method]:<6} "
                  f"{rankings['Utilization'][method]:<6} {rankings['Throughput'][method]:<11} "
                  f"{rankings['Adherence'][method]:<10} {score:<15}")
        
        # Highlight performance categories
        print(f"\n{'='*60}")
        print("PERFORMANCE CATEGORY ANALYSIS")
        print(f"{'='*60}")
        
        advanced_rl_scores = [score for method, score in overall_ranking if method in self.advanced_rl]
        basic_rl_scores = [score for method, score in overall_ranking if method in self.basic_rl]
        baseline_scores = [score for method, score in overall_ranking if method in self.baseline_schedulers]
        
        print(f"Advanced RL Average Score: {np.mean(advanced_rl_scores):.2f}")
        print(f"Basic RL Average Score: {np.mean(basic_rl_scores):.2f}")
        print(f"Baseline Schedulers Average Score: {np.mean(baseline_scores):.2f}")
        
        # Show top performers
        print(f"\n{'='*40}")
        print("TOP PERFORMERS BY CATEGORY")
        print(f"{'='*40}")
        
        advanced_rl_top = min([(method, score) for method, score in overall_ranking if method in self.advanced_rl])
        basic_rl_top = min([(method, score) for method, score in overall_ranking if method in self.basic_rl])
        baseline_top = min([(method, score) for method, score in overall_ranking if method in self.baseline_schedulers])
        
        print(f"Best Advanced RL: {advanced_rl_top[0]} (Score: {advanced_rl_top[1]})")
        print(f"Best Basic RL: {basic_rl_top[0]} (Score: {basic_rl_top[1]})")
        print(f"Best Baseline: {baseline_top[0]} (Score: {baseline_top[1]})")
        
    def create_statistical_analysis(self):
        """Create statistical analysis comparing algorithm categories"""
        print(f"\n{'='*60}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*60}")
        
        # Extract performance metrics by category
        advanced_rl_methods = [method for method in self.results.keys() if method in self.advanced_rl]
        basic_rl_methods = [method for method in self.results.keys() if method in self.basic_rl]
        baseline_methods = [method for method in self.results.keys() if method in self.baseline_schedulers]
        
        # Calculate category statistics
        categories = ['Advanced RL', 'Basic RL', 'Baseline Schedulers']
        method_groups = [advanced_rl_methods, basic_rl_methods, baseline_methods]
        
        metrics = ['cost', 'avg_time', 'utilization', 'throughput', 'adherence']
        
        for metric in metrics:
            print(f"\n{metric.upper()} ANALYSIS:")
            print("-" * 40)
            
            for category, methods in zip(categories, method_groups):
                if methods:
                    values = [self.results[method][metric] for method in methods]
                    print(f"{category}:")
                    print(f"  Mean: {np.mean(values):.4f}")
                    print(f"  Std: {np.std(values):.4f}")
                    print(f"  Min: {np.min(values):.4f}")
                    print(f"  Max: {np.max(values):.4f}")
                    
    def save_comprehensive_results(self):
        """Save comprehensive results"""
        summary_file = os.path.join(self.output_dir, 'comprehensive_comparison_summary.csv')
        
        with open(summary_file, 'w', newline='') as file:
            writer = csv.writer(file)
            methods = list(self.results.keys())
            writer.writerow(['Method', 'Category'] + ['Cost', 'Avg Time', 'Utilization', 'Throughput', 'Adherence', 'Good Placement', 'Training Time'])
            
            for method in methods:
                if method in self.advanced_rl:
                    category = 'Advanced RL'
                elif method in self.basic_rl:
                    category = 'Basic RL'
                else:
                    category = 'Baseline Scheduler'
                
                row = [method, category] + [
                    self.results[method]['cost'],
                    self.results[method]['avg_time'],
                    self.results[method]['utilization'],
                    self.results[method]['throughput'],
                    self.results[method]['adherence'],
                    self.results[method]['good_placement'],
                    self.results[method].get('training_time', 0)
                ]
                writer.writerow(row)
        
        print(f"\nComprehensive results saved to: {summary_file}")
        
    def run_complete_comparison(self, num_iterations=10000):
        """Run complete comparison"""
        print("COMPREHENSIVE COMPARISON: Basic RL vs Advanced RL vs Baseline Schedulers")
        print("=" * 100)
        
        # Setup
        self.setup_output_directory()
        self.initialize_environment()
        
        # Run Basic RL algorithms
        print("\n" + "="*60)
        print("TRAINING BASIC RL ALGORITHMS")
        print("="*60)
        for algorithm in self.basic_rl:
            try:
                results = self.run_rl_algorithm(algorithm, num_iterations)
                if results:
                    self.results[algorithm] = results
            except Exception as e:
                print(f"Failed to run {algorithm}: {str(e)}")
                continue
        
        # Run Advanced RL algorithms
        print("\n" + "="*60)
        print("TRAINING ADVANCED RL ALGORITHMS")
        print("="*60)
        for algorithm in self.advanced_rl:
            try:
                results = self.run_rl_algorithm(algorithm, num_iterations)
                if results:
                    self.results[algorithm] = results
            except Exception as e:
                print(f"Failed to run {algorithm}: {str(e)}")
                continue
        
        # Run baseline schedulers
        try:
            baseline_results = self.run_baseline_schedulers()
            self.results.update(baseline_results)
        except Exception as e:
            print(f"Failed to run baseline schedulers: {str(e)}")
        
        # Generate comprehensive comparisons
        if self.results:
            self.create_comprehensive_comparison_plots()
            self.save_comprehensive_results()
            
            print(f"\n{'='*100}")
            print("COMPREHENSIVE COMPARISON COMPLETE!")
            print(f"{'='*100}")
            print(f"Results saved to: {self.output_dir}")
            print(f"Total methods compared: {len(self.results)}")
            
            # Print final summary
            advanced_rl_methods = [m for m in self.results.keys() if m in self.advanced_rl]
            basic_rl_methods = [m for m in self.results.keys() if m in self.basic_rl]
            baseline_methods = [m for m in self.results.keys() if m in self.baseline_schedulers]
            
            print(f"\nAdvanced RL Methods: {advanced_rl_methods}")
            print(f"Basic RL Methods: {basic_rl_methods}")
            print(f"Baseline Schedulers: {baseline_methods}")
            
        else:
            print("No results to compare!")

# Main execution
def main():
    print("Comprehensive Comparison: Basic RL vs Advanced RL vs Baseline Schedulers")
    print("=" * 100)
    
    manager = ComprehensiveComparisonManager()
    
    try:
        iterations = int(input("Enter number of iterations for RL algorithms (default 10000): ") or "10000")
    except ValueError:
        iterations = 10000
    
    manager.run_complete_comparison(num_iterations=iterations)

if __name__ == "__main__":
    main()
