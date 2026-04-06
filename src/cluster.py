import numpy as np
import definitions as defs
import workload
import copy
import constants

# cluster resource details


vm_types = 3

vm1_total = 3
vm1_cpu = 4
vm1_mem = 12
vm1_price = 1

vm2_total = 3
vm2_cpu = 8
vm2_mem = 24
vm2_price = 2

vm3_total = 3
vm3_cpu = 12
vm3_mem = 36
vm3_price = 3

j_total = 3
j_types = 3
j_cpu_max = 4
j_cpu_min = 1
j_mem_max = 12
j_mem_min = 1
j_ex_max = 8
j_ex_min = 1

j1_cpu = 2
j1_mem = 4
j1_ex = 3
j1_time = 60

j2_cpu = 4
j2_mem = 8
j2_ex = 2
j2_time = 100

j3_cpu = 6
j3_mem = 8
j3_ex = 2
j3_time = 80

job_features = 4
features = (vm1_total + vm2_total + vm3_total) * 2 + (job_features * constants.job_batch_size)
# cluster_state_min = [0, 0, 0, 0, 0, 0, 1, 1, j_cpu_min, j_mem_min, j_ex_min]
# cluster_state_max = [vm0_cpu, vm0_mem, vm1_cpu, vm1_mem, vm2_cpu, vm2_mem, j_total, j_types, j_cpu_max, j_mem_max, j_ex_max]
# cluster_state_init = [vm0_cpu, vm0_mem, vm1_cpu, vm1_mem, vm2_cpu, vm2_mem, 1, 1, j0_cpu, j0_mem, j0_ex]

JOBS = []
VMS = []

# cluster_state_init = []
cluster_state_init = []
cluster_state_min = []
cluster_state_max = []
# job_queue = PriorityQueue()

max_episode_cost = 0
min_avg_job_duration = 0


def gen_cluster_state(job_idx, jobs, vms):
    """Generate cluster state using VM resources and next N job features."""
    cluster_state = []
    i = 0
    while i < len(vms):
        cluster_state.append(vms[i].cpu_now)
        cluster_state.append(vms[i].mem_now)
        i += 1
    batch_size = max(1, int(constants.job_batch_size))
    for offset in range(batch_size):
        idx = job_idx + offset
        if idx < len(jobs):
            cluster_state.append(jobs[idx].type)
            cluster_state.append(jobs[idx].cpu)
            cluster_state.append(jobs[idx].mem)
            cluster_state.append(jobs[idx].ex - jobs[idx].ex_placed)
        else:
            # Zero-pad missing jobs so shape stays fixed for TF-Agents.
            cluster_state.extend([0, 0, 0, 0])
    return cluster_state


def gen_cluster_state_min():
    cluster_state = []
    i = 0
    while i < len(VMS):
        cluster_state.append(0)
        cluster_state.append(0)
        i += 1
    batch_size = max(1, int(constants.job_batch_size))
    for _ in range(batch_size):
        cluster_state.append(0)
        cluster_state.append(0)
        cluster_state.append(0)
        cluster_state.append(0)
    return cluster_state


def gen_cluster_state_max():
    cluster_state = []
    i = 0
    while i < len(VMS):
        cluster_state.append(VMS[i].cpu)
        cluster_state.append(VMS[i].mem)
        i += 1
    batch_size = max(1, int(constants.job_batch_size))
    for _ in range(batch_size):
        cluster_state.append(j_types)
        cluster_state.append(j_cpu_max)
        cluster_state.append(j_mem_max)
        cluster_state.append(j_ex_max)
    return cluster_state


def gen_jobs_simple():
    global JOBS
    JOBS = []
    JOBS.append(defs.JOB(0, 0, 1, j1_cpu, j1_mem, j1_ex, j1_time))
    JOBS.append(defs.JOB(50, 1, 2, j2_cpu, j2_mem, j2_ex, j2_time))
    JOBS.append(defs.JOB(80, 2, 3, j3_cpu, j3_mem, j3_ex, j3_time))


def fetch_jobs_workload():
    global JOBS
    JOBS = copy.deepcopy(workload.JOBS_WORKLOAD)


def init_jobs():
    # gen_jobs_simple()
    fetch_jobs_workload()


def init_vms():
    global VMS
    VMS = []
    for i in range(vm1_total):
        VMS.append(defs.VM(len(VMS), vm1_cpu, vm1_mem, vm1_price))
    for j in range(vm2_total):
        VMS.append(defs.VM(len(VMS), vm2_cpu, vm2_mem, vm2_price))
    for k in range(vm3_total):
        VMS.append(defs.VM(len(VMS), vm3_cpu, vm3_mem, vm3_price))


def update_vm_prices(vms):
    """Apply per-step stochastic VM pricing when enabled."""
    if not constants.enable_stochastic_pricing:
        for vm in vms:
            vm.current_price = vm.base_price
        return

    for vm in vms:
        vm.current_price = max(
            0.1,
            vm.base_price + np.random.normal(0, constants.price_variance)
        )


def init_cluster():
    init_jobs()
    init_vms()
    global features
    features = (vm1_total + vm2_total + vm3_total) * 2 + (job_features * max(1, int(constants.job_batch_size)))
    global cluster_state_init
    global cluster_state_max
    global cluster_state_min
    cluster_state_init = gen_cluster_state(0, JOBS, VMS)
    cluster_state_min = gen_cluster_state_min()
    cluster_state_max = gen_cluster_state_max()

    vms_total_price = 0
    jobs_total_time = 0

    global max_episode_cost
    global min_avg_job_duration
    global total_cpu  # add this line
    global total_cpu_used  # add this line
    total_cpu = 0  # add this line
    total_cpu_used = 0  # add this line
    for i in range(len(VMS)):
        vms_total_price += VMS[i].price
        total_cpu += VMS[i].cpu  # add this line
    for i in range(len(JOBS)):
        jobs_total_time += JOBS[i].duration

    max_episode_cost = vms_total_price * jobs_total_time
    min_avg_job_duration = float(jobs_total_time) / len(JOBS)

    # print('Number of VMS {0}'.format(len(VMS)))
    # print('Number of Jobs {0}'.format(len(JOBS)))
    # print('MAX Episode VM Cost {0}'.format(total_episode_cost))
    # print('MIN Episode AVG Job Duration {0}'.format(min_avg_job_duration))
