import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# NOTE: prep.preprocess is a function that extracts machine assignments,
#       processing and waiting times, due-dates and job lengths from the
#       raw data. It is not included as it is tailored to a specific
#       format of the data. Job data is returned as 2d array, where the
#       first dimension iterates over jobs and the second over operations.


### Preprocessing ###

def adjust_deadlines(p, w, job_deadlines):
    """
    Modifies the deadlines to take into account the absence of night-time in the model.
    Also ensures deadlines are not shorter than the jobs.
    Should be applied after the adjustment of waiting times (if used). 
    """

    for j in range(len(job_deadlines)):
        processing = sum(p[j]) + sum(w[j])
        if processing > job_deadlines[j]/3:
            job_deadlines[j] = processing
        else:
            job_deadlines[j] = job_deadlines[j]/3

    return job_deadlines


def adjust_waiting_times(w, job_lengths):
    """
    Reduces the waiting times to account for night-time.
    """

    w_new = []

    for j in range(len(job_lengths)):
        tmp = []
        for o in range(job_lengths[j]):
            real_w = 0
            while w[j][o] > 0:
                real_w += min(8*60, w[j][o])
                w[j][o] = w[j][o] - 24*60
            tmp.append(real_w)
        w_new.append(tmp)

    return w_new


### Tests and comparisons ###

def compare_greedy_algos():

    job_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/orders.csv"
    operation_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/tasks.csv"
    json_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/data.json"

    m, p, w, d, jl, m_dict, j_dict = prep.preprocess(
        job_file, operation_file, json_file)

    w = adjust_waiting_times(w, jl)
    d = adjust_deadlines(p, w, d)

    f = open("greedy_methods.csv", "w")
    f.write("optimal;simple;local;remainingWork;relaxedMachines;forcedMachines\n")
    # f.write("optimal;simple;remainingWork;relaxedMachines;forcedMachines\n")
    f.close()

    for i in range(100):
        ma, pt, wt, dl, job_lengths, indices = sample(
            m, p, w, d, jl, 50)

        opt = 0
        simple = 0
        local = 0
        remWork = 0
        relax = 0
        forced = 0

        # Standard solve
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        sol, mksp = simple_mksp_greedy(model)
        model = set_big_M_constraints(model, mksp)
        # model = set_max_tardiness_objective(model, dl)
        # sol, _, tard = simple_tard_greedy(model)
        # model = set_big_M_constraints(model, tard)
        model = set_initial_solution(model, sol)
        model.Params.MIPFocus = 2
        model.Params.IntegralityFocus = 1
        model.optimize()
        opt = model.ObjVal

        """ Only for makespan """

        # Local
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        sol, mksp = simple_mksp_greedy(model)
        simple = mksp
        model = set_initial_solution(model, sol)
        model = set_big_M_constraints(model, mksp)
        model = adapt_for_greedy(model, 50, "local")
        error = False
        try:
            model = large_step_greedy(model)
        except:
            error = True

        if error or model.status == 3:
            local = -1
        else:
            local = model.ObjVal

        # Remaining work
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        sol, mksp = simple_mksp_greedy(model)
        # model = set_tot_tardiness_objective(model, dl)
        # sol, _, tard = simple_tard_greedy(model)
        # simple = tard
        model = set_initial_solution(model, sol)
        model = set_big_M_constraints(model, mksp)
        # model = set_big_M_constraints(model, tard)
        model = adapt_for_greedy(model, 50, "remainingWork")
        error = False
        try:
            model = large_step_greedy(model)
        except:
            error = True

        if error or model.status == 3:
            remWork = -1
        else:
            remWork = model.ObjVal

        # Relaxed machines
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        sol, mksp = simple_mksp_greedy(model)
        # model = set_tot_tardiness_objective(model, dl)
        # sol, _, tard = simple_tard_greedy(model)
        model = set_initial_solution(model, sol)
        model = set_big_M_constraints(model, mksp)
        # model = set_big_M_constraints(model, tard)
        model = adapt_for_greedy(model, 50)
        error = False
        try:
            model = large_step_greedy(model)
        except:
            error = True

        if error or model.status == 3:
            relax = -1
        else:
            relax = model.ObjVal

        # Forced machines
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        sol, mksp = simple_mksp_greedy(model)
        # model = set_tot_tardiness_objective(model, dl)
        # sol, _, tard = simple_tard_greedy(model)
        model = set_initial_solution(model, sol)
        model = set_big_M_constraints(model, mksp)
        # model = set_big_M_constraints(model, tard)
        model = adapt_for_greedy(model, 50)
        model = set_before_forcing(model)
        error = False
        try:
            model = large_step_greedy(model)
        except:
            error = True

        if error or model.status == 3:
            forced = -1
        else:
            forced = model.ObjVal

        f = open("greedy_methods.csv", "a")
        f.write(f"{opt};{simple};{local};{remWork};{relax};{forced}\n")
        # f.write(f"{opt};{simple};{remWork};{relax};{forced}\n")
        f.close()


def compare_big_M_methods():

    job_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/orders.csv"
    operation_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/tasks.csv"
    json_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/data.json"

    m, p, w, d, jl, m_dict, j_dict = prep.preprocess(
        job_file, operation_file, json_file)

    w = adjust_waiting_times(w, jl)
    d = adjust_deadlines(p, w, d)

    f = open("big_M_methods.csv", "w")
    f.write("standard;binary;variable\n")
    f.close()

    for i in range(100):
        ma, pt, wt, dl, job_lengths, indices = sample(
            m, p, w, d, jl, 50)

        # Standard solve
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        model.Params.MIPFocus = 2
        model.Params.IntegralityFocus = 1

        tic_std = time.time()
        sol, u_bound = simple_mksp_greedy(model)
        model = set_big_M_constraints(model, u_bound)
        model.Params.TimeLimit = 80
        model.optimize()
        toc_std = time.time()

        if toc_std-tic_std > 60:
            continue

        # Binary solve
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        model.Params.MIPFocus = 2
        model.Params.IntegralityFocus = 1

        tic_bin = time.time()
        model = binary_M_solve(model)
        toc_bin = time.time()

        # Variable M solve
        model = build_disjunctive_model(ma, pt, wt, job_lengths)
        model = set_makespan_objective(model)
        model.Params.MIPFocus = 2
        model.Params.IntegralityFocus = 1

        tic_var = time.time()
        model = variable_big_M_solve(model)
        toc_var = time.time()

        f = open("big_M_methods.csv", "a")
        f.write(f"{toc_std-tic_std};{toc_bin-tic_bin};{toc_var-tic_var}\n")
        f.close()


def test_breakdown():

    job_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/orders.csv"
    operation_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/tasks.csv"
    json_file = "C:/Users/romet/Desktop/Thesis/datasets/2022_06/data.json"

    m, p, w, d, jl, m_dict, j_dict = prep.preprocess(
        job_file, operation_file, json_file)

    w = adjust_waiting_times(w, jl)
    d = adjust_deadlines(p, w, d)

    ma, pt, wt, dl, job_lengths, indices = sample(
        m, p, w, d, jl, 100)

    model = build_disjunctive_model(ma, pt, wt, job_lengths)
    model = set_max_tardiness_objective(model, dl)
    # model.Params.LogToConsole = 0
    model.Params.MIPFocus = 2
    model.Params.IntegralityFocus = 1
    sol, mksp = simple_tard_greedy(model)
    model = set_big_M_constraints(model, mksp)
    model = set_initial_solution(model, sol)
    model.optimize()

    init_op = model.objVal

    model, job, operation, delay = simulate_breakdown(model)
    model = reset_indicator_constraints(model)

    R1, fix1 = right_shift_repair(model, job, operation, delay)
    R2, fix2 = affected_operations_repair(model, job, operation, delay)

    tard1 = 0
    tard2 = 0
    for j in range(model._n_jobs):
        tard1 = max(tard1, fix1[j][model._job_lengths[j]-1] +
                    model._pt[j][model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1] - dl[j])
        tard2 = max(tard2, fix2[j][model._job_lengths[j]-1] +
                    model._pt[j][model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1] - dl[j])

    model = set_initial_solution(model, fix2)
    model.optimize()

    print(f"Right shift repair: {tard1}")
    print(f"AO repair:          {tard2}")
    print(f"Optimal repair:     {model.objVal}")


### Model building ###

def sample(ma, pt, wt, dl, job_lengths, size, rng_seed=0):

    n_jobs = len(job_lengths)

    if rng_seed != 0:
        rng = np.random.default_rng(seed=rng_seed)
    else:
        rng = np.random.default_rng()

    indices = rng.choice(n_jobs, size, False)

    sample_ma = []
    sample_pt = []
    sample_wt = []
    sample_dl = []
    sample_jl = []

    for i in indices:
        sample_ma.append(ma[i])
        sample_pt.append(pt[i])
        sample_wt.append(wt[i])
        sample_dl.append(dl[i])
        sample_jl.append(job_lengths[i])

    return sample_ma, sample_pt, sample_wt, sample_dl, sample_jl, indices


def build_disjunctive_model(ma, pt, wt, job_lengths):
    """
    Builds and returns a disjunctive model from the given machine assignment, processing times and waiting times.
    """

    m = gp.Model("disjunctive_JSSP")

    m._ma = ma
    m._pt = pt
    m._wt = wt
    m._job_lengths = job_lengths
    m._n_jobs = len(job_lengths)
    m._n_machines = 0
    for j in range(m._n_jobs):
        m._n_machines = max(m._n_machines, max(ma[j]))
    m._n_machines += 1

    # Variables
    m._start = m.addVars([(j, o) for j in range(m._n_jobs) for o in range(
        m._job_lengths[j])], vtype=GRB.INTEGER, name="start")
    m._before = m.addVars([(j, o, k, q) for j in range(m._n_jobs) for o in range(m._job_lengths[j]) for k in range(
        j+1, m._n_jobs) for q in range(m._job_lengths[k]) if (m._ma[j][o] == m._ma[k][q])], vtype=GRB.BINARY, name="before")

    # Constraints
    m._prec = m.addConstrs((m._start[j, o+1] >= m._start[j, o] + m._pt[j][o] + m._wt[j][o]
                           for j in range(m._n_jobs) for o in range(m._job_lengths[j]-1)), name="precedence")

    m._machine1 = m.addConstrs((((m._before[j, o, k, q] == 1) >> (m._start[k, q] >= m._start[j, o] + m._pt[j][o])) for j in range(m._n_jobs) for o in range(
        m._job_lengths[j]) for k in range(j+1, m._n_jobs) for q in range(m._job_lengths[k]) if (m._ma[j][o] == m._ma[k][q])), name="machine1")
    m._machine2 = m.addConstrs((((m._before[j, o, k, q] == 0) >> (m._start[j, o] >= m._start[k, q] + m._pt[k][q])) for j in range(m._n_jobs) for o in range(
        m._job_lengths[j]) for k in range(j+1, m._n_jobs) for q in range(m._job_lengths[k]) if (m._ma[j][o] == m._ma[k][q])), name="machine2")

    return m


def set_big_M_constraints(model, big_M, pairwise=False, bound_mksp=False):

    model._big_M = big_M
    model.update()
    model.remove(model._machine1)
    model.remove(model._machine2)

    if hasattr(model, "_tardiness"):

        if pairwise:

            if not hasattr(model, "_head"):
                model = compute_heads_tails(model)

            model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o] - (model._dl[j]+big_M+model._pt[j][o]-model._head[k][q]-model._tail[j][o])*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
            model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q] - (model._dl[k]+big_M+model._pt[k][q]-model._head[j][o]-model._tail[k][q])*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

        else:
            model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o] - (big_M+model._dl[j])*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
            model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q] - (big_M+model._dl[k])*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

    elif hasattr(model, "_makespan"):

        if pairwise:

            if not hasattr(model, "_head"):
                model = compute_heads_tails(model)

            model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o] - (big_M+model._pt[j][o]-model._head[k][q]-model._tail[j][o])*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
            model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q] - (big_M+model._pt[k][q]-model._head[j][o]-model._tail[k][q])*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

        else:
            model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o] - big_M*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
            model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q] - big_M*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
                model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

        if bound_mksp:
            model._mksp_M_bound = model.addConstr(
                model._makespan <= big_M, name="makespan_M_bound")

    return model


def set_makespan_objective(model):
    """
    Returns the model with additional variables and constraints for minimizing the makespan objective.
    """

    model._makespan = model.addVar(obj=1, vtype=GRB.INTEGER, name="makespan")

    model._mksp_bound = model.addConstrs((
        (model._makespan >= model._start[j, model._job_lengths[j]-1] + model._pt[j][model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1]) for j in range(model._n_jobs)), name="makespan_bound")

    return model


def set_tot_tardiness_objective(model, dl):

    model._dl = dl
    model._tardiness = model.addVars(
        [(j) for j in range(model._n_jobs)], obj=1, vtype=GRB.INTEGER, name="tardiness")

    model._tard_bound = model.addConstrs(((model._tardiness[j] >= model._start[j, model._job_lengths[j]-1] + model._pt[j]
                                         [model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1] + - dl[j]) for j in range(model._n_jobs)), name="tardiness_bound")

    return model


def set_max_tardiness_objective(model, dl):

    model._dl = dl
    model._tardiness = model.addVars(
        [(j) for j in range(model._n_jobs)], lb=-float('inf'), vtype=GRB.INTEGER, name="tardiness")

    model._tard_bound = model.addConstrs(((model._tardiness[j] >= model._start[j, model._job_lengths[j]-1] + model._pt[j]
                                         [model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1] - dl[j]) for j in range(model._n_jobs)), name="tardiness_bound")

    model._max_tardiness = model.addVar(
        obj=1, lb=-float('inf'), vtype=GRB.INTEGER, name="max_tardiness")

    model._max_tard_bound = model.addConstrs(
        ((model._max_tardiness >= model._tardiness[j]) for j in range(model._n_jobs)), name="max_tardiness_bound")

    return model


def set_tot_lateness_objective(model, dl):

    model._dl = dl
    model._tardiness = model.addVars(
        [(j) for j in range(model._n_jobs)], lb=-float('inf'), obj=1, vtype=GRB.INTEGER, name="lateness")

    model._late_bound = model.addConstrs(((model._tardiness[j] >= model._start[j, model._job_lengths[j]-1] + model._pt[j]
                                         [model._job_lengths[j]-1] + model._wt[j][model._job_lengths[j]-1] - dl[j]) for j in range(model._n_jobs)), name="lateness_bound")

    return model


def set_initial_solution(model, sol, fix=False):
    """
    Sets the initial solution of the model according to the given starting times.
    """

    for j in range(model._n_jobs):
        for o in range(model._job_lengths[j]):
            model._start[j, o].Start = sol[j][o]
            if fix:
                model._start[j, o].lb = sol[j][o]
                model._start[j, o].ub = sol[j][o]
            for k in range(j+1, model._n_jobs):
                for q in range(model._job_lengths[k]):
                    if model._ma[j][o] == model._ma[k][q]:
                        if sol[j][o] < sol[k][q]:
                            model._before[j, o, k, q].Start = 1
                        else:
                            model._before[j, o, k, q].Start = 0

    if hasattr(model, "_dl"):
        for j in range(model._n_jobs):
            model._tardiness[j].Start = max(
                0, sol[j][model._job_lengths[j]-1]+model._pt[j][model._job_lengths[j]-1]+model._wt[j][model._job_lengths[j]-1]-model._dl[j])

    return model


### Optimization methods ###

def binary_M_solve(model):

    if hasattr(model, "_makespan"):
        sol, u_bound = simple_mksp_greedy(model)
    elif hasattr(model, "_tardiness"):
        sol, u_bound = simple_tard_greedy(model)

    if hasattr(model, "_makespan"):
        job_bound = np.zeros(model._n_jobs)
        machine_bound = np.zeros(model._n_machines)
        for j in range(model._n_jobs):
            for o in range(model._job_lengths[j]):
                job_bound[j] += model._pt[j][o] + model._wt[j][o]
                machine_bound[model._ma[j][o]] += model._pt[j][o]
        l_bound = max(max(job_bound), max(machine_bound))
    elif hasattr(model, "_tardiness"):
        l_bound = 0

    M = np.floor((l_bound+u_bound)/2)
    model = set_initial_solution(model, sol)
    model = set_big_M_constraints(model, M, bound_mksp=True)
    model.Params.BestObjStop = GRB.INFINITY

    while True:
        model.optimize()
        if model.status == 3:
            l_bound = M+1
        else:
            u_bound = M

        if l_bound == u_bound:
            return model
        M = np.floor((l_bound + u_bound)/2)

        model = set_big_M_constraints(model, M, bound_mksp=True)


def variable_big_M_solve(model):

    if hasattr(model, "_makespan"):
        sol, u_bound = simple_mksp_greedy(model)
    elif hasattr(model, "_tardiness"):
        sol, u_bound = simple_tard_greedy(model)

    if hasattr(model, "_makespan"):
        job_bound = np.zeros(model._n_jobs)
        machine_bound = np.zeros(model._n_machines)
        for j in range(model._n_jobs):
            for o in range(model._job_lengths[j]):
                job_bound[j] += model._pt[j][o] + model._wt[j][o]
                machine_bound[model._ma[j][o]] += model._pt[j][o]
        l_bound = max(max(job_bound), max(machine_bound))
    elif hasattr(model, "_tardiness"):
        l_bound = 0

    model = set_initial_solution(model, sol)

    model = set_big_M_constraints(model, u_bound)

    model.Params.BestObjStop = (l_bound + u_bound)/2
    model.optimize()

    while model.status != 2:
        l_bound = max(l_bound, model.ObjBound)
        model = set_big_M_constraints(model, model.ObjVal)
        model.Params.BestObjStop = (model.ObjVal + l_bound)/2
        model.optimize()

    return model


### Greedy methods ###

def simple_mksp_greedy(model):
    """
    Applies a greedy algorithm to produce a schedule with (hopefully) a short makespan. Returns the starting times and makespan of said scheduled.
    """

    # Note: no need to update the machine bound for the new active task

    active_indices = np.zeros(model._n_jobs, dtype=int)
    start_times = [[0 for o in range(model._job_lengths[j])]
                   for j in range(model._n_jobs)]
    mksp = 0

    next_pt = np.zeros(model._n_jobs)
    next_wt = np.zeros(model._n_jobs)
    next_ma = np.zeros(model._n_jobs)
    for j in range(model._n_jobs):
        next_pt[j] = model._pt[j][0]
        next_wt[j] = model._wt[j][0]
        next_ma[j] = model._ma[j][0]

    job_bound = np.zeros(model._n_jobs)
    machine_bound = np.zeros(model._n_jobs)

    finished = 0
    while finished < model._n_jobs:

        EST = np.maximum(job_bound, machine_bound)
        j = np.argmin(EST+next_pt+next_wt)

        start_times[j][active_indices[j]] = EST[j]
        mksp = max(mksp, EST[j]+next_pt[j]+next_wt[j])
        job_bound[j] = EST[j]+next_pt[j]+next_wt[j]
        machine_bound[next_ma == model._ma[j]
                      [active_indices[j]]] = EST[j]+next_pt[j]

        active_indices[j] += 1
        if active_indices[j] == model._job_lengths[j]:
            next_pt[j] = np.inf
            next_wt[j] = np.inf
            finished += 1
        else:
            next_pt[j] = model._pt[j][active_indices[j]]
            next_wt[j] = model._wt[j][active_indices[j]]
            next_ma[j] = model._ma[j][active_indices[j]]

    return start_times, mksp


def simple_tard_greedy(model):
    """
    Applies a greedy algorithm to produce a schedule with (hopefully) little tardiness. Returns the starting times, total tardiness and max tardiness of said scheduled.
    """

    active_indices = np.zeros(model._n_jobs, dtype=int)
    start_times = [[0 for o in range(model._job_lengths[j])]
                   for j in range(model._n_jobs)]
    tot_tardiness = 0
    max_tardiness = -np.inf

    next_pt = np.zeros(model._n_jobs)
    next_wt = np.zeros(model._n_jobs)
    next_startdl = np.zeros(model._n_jobs)
    next_ma = np.zeros(model._n_jobs)
    for j in range(model._n_jobs):
        next_pt[j] = model._pt[j][0]
        next_wt[j] = model._wt[j][0]
        next_startdl[j] = model._dl[j] - sum(model._pt[j]) - sum(model._wt[j])
        next_ma[j] = model._ma[j][0]

    job_bound = np.zeros(model._n_jobs)
    machine_bound = np.zeros(model._n_jobs)
    machine_free = np.zeros(model._n_machines)

    finished = 0
    while finished < model._n_jobs:

        EST = np.maximum(job_bound, machine_bound)
        obj = next_startdl-EST
        obj[EST != min(EST)] = np.inf
        j = np.argmin(obj)

        start_times[j][active_indices[j]] = EST[j]
        job_bound[j] = EST[j]+next_pt[j]+next_wt[j]
        machine_bound[next_ma == model._ma[j]
                      [active_indices[j]]] = EST[j]+next_pt[j]
        machine_free[model._ma[j][active_indices[j]]] = EST[j]+next_pt[j]

        active_indices[j] += 1
        if active_indices[j] == model._job_lengths[j]:
            tot_tardiness += max(0, EST[j] + next_pt[j] +
                                 next_wt[j] - model._dl[j])
            max_tardiness = max(max_tardiness, EST[j] + next_pt[j] +
                                next_wt[j] - model._dl[j])
            next_pt[j] = 0
            next_wt[j] = 0
            next_startdl[j] = 0
            job_bound[j] = np.inf
            next_ma[j] = np.nan
            finished += 1
        else:
            next_startdl[j] += next_pt[j] + next_wt[j]
            next_pt[j] = model._pt[j][active_indices[j]]
            next_wt[j] = model._wt[j][active_indices[j]]
            next_ma[j] = model._ma[j][active_indices[j]]
            machine_bound[j] = machine_free[int(next_ma[j])]

    return start_times, tot_tardiness, max_tardiness


def large_step_greedy(model, activation=False, bound_unscheduled=False):

    model.Params.TimeLimit = 120

    n_ops = sum(model._job_lengths)
    if bound_unscheduled:
        machine_free = np.zeros(model._n_machines)

    while True:
        model.optimize()

        if activation:
            active = semi_activate(model)

        for j in range(model._n_jobs):
            for o in range(model._job_lengths[j]):
                if model._scheduled[j, o].X >= .9:
                    model._scheduled[j, o].lb = 1
                    if activation:
                        model._start[j, o].lb = active[j][o]
                        model._start[j, o].ub = active[j][o]
                        if bound_unscheduled:
                            machine_free[model._ma[j][o]] = max(
                                machine_free[model._ma[j][o]], active[j][o] + model._pt[j][o])
                    else:
                        model._start[j, o].lb = model._start[j, o].X
                        model._start[j, o].ub = model._start[j, o].X
                        if bound_unscheduled:
                            machine_free[model._ma[j][o]] = max(
                                machine_free[model._ma[j][o]], model._start[j, o].X + model._pt[j][o])

        if bound_unscheduled:
            for j in range(model._n_jobs):
                for o in range(model._job_lengths[j]):
                    if model._scheduled[j, o].X < .1:
                        model._start[j, o].lb = machine_free[model._ma[j][o]]

        # print_line()
        # print(f"Scheduled {model._stepConstr.RHS} operations (of {n_ops}).")

        if model._stepConstr.RHS == n_ops:
            return model

        model._stepConstr.RHS = min(model._stepConstr.RHS + model._step, n_ops)


def adapt_for_greedy(model, step, method="relaxed"):

    model._step = step

    model._scheduled = model.addVars(((j, o) for j in range(model._n_jobs) for o in range(
        model._job_lengths[j])), vtype=GRB.BINARY, name="scheduled")

    model._stepConstr = model.addConstr(gp.quicksum(
        model._scheduled) >= step, name="step")

    model._partial = model.addConstrs(((model._scheduled[j, o+1] <= model._scheduled[j, o])
                                      for j in range(model._n_jobs) for o in range(model._job_lengths[j]-1)), name="partial")

    model.update()

    # Update machine constraints
    model.remove(model._machine1)
    model.remove(model._machine2)

    if hasattr(model, "_tardiness"):
        model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o]*model._scheduled[j, o] - (model._big_M+model._dl[j])*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
            model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
        model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q]*model._scheduled[k, q] - (model._big_M+model._dl[k])*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
            model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

    elif hasattr(model, "_makespan"):
        model._machine1 = model.addConstrs(((model._start[k, q] >= model._start[j, o] + model._pt[j][o]*model._scheduled[j, o] - model._big_M*(1-model._before[j, o, k, q])) for j in range(model._n_jobs) for o in range(
            model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
        model._machine2 = model.addConstrs(((model._start[j, o] >= model._start[k, q] + model._pt[k][q]*model._scheduled[k, q] - model._big_M*model._before[j, o, k, q]) for j in range(model._n_jobs) for o in range(
            model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

    if method == "local":
        # Update objective and precedence
        model.remove(model._prec)
        model._prec = model.addConstrs((model._start[j, o+1] >= model._start[j, o] + (model._pt[j][o] + model._wt[j][o])
                                       * model._scheduled[j, o] for j in range(model._n_jobs) for o in range(model._job_lengths[j]-1)), name="precedence")

        if hasattr(model, "_makespan"):
            model.remove(model._mksp_bound)
            model._mksp_bound = model.addConstrs((
                (model._makespan >= model._start[j, o] + (model._pt[j][o] + model._wt[j][o])*model._scheduled[j, o]) for j in range(model._n_jobs) for o in range(model._job_lengths[j])), name="makespan_bound")

        if hasattr(model, "_tardiness"):
            raise NotImplementedError(
                "For tardiness objective: Local equivalent to remainingWork")

    if method == "remainingWork":

        model = compute_heads_tails(model)
        # Update objective and precedence
        model.remove(model._prec)
        model._prec = model.addConstrs((model._start[j, o+1] >= model._start[j, o] + (model._pt[j][o] + model._wt[j][o])
                                       * model._scheduled[j, o] for j in range(model._n_jobs) for o in range(model._job_lengths[j]-1)), name="precedence")

        if hasattr(model, "_makespan"):
            model.remove(model._mksp_bound)
            model._mksp_bound = model.addConstrs((
                (model._makespan >= model._start[j, o] + model._tail[j][o]) for j in range(model._n_jobs) for o in range(model._job_lengths[j])), name="makespan_bound")

        if hasattr(model, "_tardiness"):
            model.remove(model._tard_bound)
            model._tard_bound = model.addConstrs(((model._tardiness[j] >= model._start[j, o] + model._tail[j][o] - model._dl[j]) for j in range(
                model._n_jobs) for o in range(model._job_lengths[j])), name="tardiness_bound")

    return model


def set_before_forcing(model):

    model._forcing1 = model.addConstrs(((model._before[j, o, k, q] >= model._scheduled[j, o] - model._scheduled[k, q]) for j in range(model._n_jobs) for o in range(
        model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if model._ma[j][o] == model._ma[k][q]), name="forcing1")

    model._forcing2 = model.addConstrs(((model._before[j, o, k, q] <= 1 + model._scheduled[j, o] - model._scheduled[k, q]) for j in range(model._n_jobs) for o in range(
        model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if model._ma[j][o] == model._ma[k][q]), name="forcing2")

    return model


### Breakdowns ###

def simulate_breakdown(model, rng_seed=0):

    if rng_seed != 0:
        rng = np.random.default_rng(seed=rng_seed)
    else:
        rng = np.random.default_rng()

    job = rng.integers(model._n_jobs)
    operation = rng.integers(model._job_lengths[job])
    delay = 15*rng.integers(1, 13)  # Delay between 15 minutes and 3 hours

    print(f"Delaying operation ({job},{operation}) by {delay} minutes.")

    # Fix operations started before (job,operation) finishes processing
    for j in range(model._n_jobs):
        for o in range(model._job_lengths[j]):
            if (j, o) != (job, operation) and model._start[j, o].X < model._start[job, operation].X + model._pt[job][operation]:
                model._start[j, o].lb = model._start[j, o].X
                model._start[j, o].ub = model._start[j, o].X
            elif (j, o) != (job, operation):
                model._start[j, o].lb = model._start[job,
                                                     operation].X + model._pt[job][operation] + delay

    # Change start of (job,operation) to account for the delay
    model._start[job, operation].lb = model._start[job, operation].X + delay
    model._start[job, operation].ub = model._start[job, operation].X + delay

    return model, job, operation, delay


def right_shift_repair(model, job, operation, delay):

    R = []
    machine_threshold = np.inf*np.ones(model._n_machines)

    for o in range(operation, model._job_lengths[job]):
        R.append((job, o))
        machine_threshold[model._ma[job][o]] = min(
            machine_threshold[model._ma[job][o]], model._start[job, o].X)

    iterate = True
    while iterate:
        iterate = False
        for j in range(model._n_jobs):
            for o in range(model._job_lengths[j]):
                if (j, o) not in R and model._start[j, o].X >= machine_threshold[model._ma[j][o]]:
                    iterate = True
                    for q in range(o, model._job_lengths[j]):
                        if (j, q) not in R:
                            R.append((j, q))
                            machine_threshold[model._ma[j][q]] = min(
                                machine_threshold[model._ma[j][q]], model._start[j, q].X)

    sol = []
    for j in range(model._n_jobs):
        tmp = []
        for o in range(model._job_lengths[j]):
            s = model._start[j, o].X
            if (j, o) in R:
                s += delay
            tmp.append(s)
        sol.append(tmp)

    return R, sol


def affected_operations_repair(model, job, operation, delay):

    sol = []
    for j in range(model._n_jobs):
        tmp = []
        for o in range(model._job_lengths[j]):
            tmp.append(model._start[j, o].X)
        sol.append(tmp)

    machine_free = np.zeros(model._n_machines)
    machine_list = [[] for _ in range(model._n_machines)]
    start_list = [[] for _ in range(model._n_machines)]
    for j in range(model._n_jobs):
        for o in range(model._job_lengths[j]):
            machine_list[model._ma[j][o]].append((j, o))
            start_list[model._ma[j][o]].append(sol[j][o])

    for m in range(model._n_machines):
        machine_list[m] = [x for _, x in sorted(
            zip(start_list[m], machine_list[m]))]

    R = []
    A = [(job, operation)]

    while len(A) != 0:
        earliest_start = np.inf
        earliest_j = -1
        earliest_o = -1
        for j, o in A:
            if sol[j][o] < earliest_start:
                earliest_start = sol[j][o]
                earliest_j = j
                earliest_o = o
        j = earliest_j
        o = earliest_o
        R.append((j, o))
        A.remove((j, o))

        m = model._ma[j][o]

        # Delay the operation
        if (j, o) == (job, operation):
            sol[j][o] += delay
        elif o == 0:
            sol[j][o] = machine_free[m]
        else:
            sol[j][o] = max(machine_free[m], sol[j][o-1] +
                            model._pt[j][o-1]+model._wt[j][o-1])

        # Update machine availability
        machine_free[m] = sol[j][o] + model._pt[j][o]

        # Check if conflict with following operation
        if o != model._job_lengths[j]-1 and sol[j][o+1] < sol[j][o] + model._pt[j][o] + model._wt[j][o]:
            if (j, o+1) not in A:
                A.append((j, o+1))

        # Check conflict on the machine
        index = machine_list[m].index((j, o))
        if index+1 < len(machine_list[m]):
            k, q = machine_list[m][index+1]
            if sol[k][q] < machine_free[m] and (k, q) not in A:
                A.append((k, q))

    return R, sol


### Utilities ###

def add_cuts(model):
    """
    Adds two-job cuts (valid inequalities) to the model.
    """

    model._E = []
    for job in range(model._n_jobs):

        tmp = [0]
        for op in range(1, model._job_lengths[job]):
            tmp.append(tmp[op-1]+model._pt[job][op-1])

        model._E.append(tmp)

    model._2job_cuts = model.addConstrs(((model._pt[k][q] + model._E[k][q] - model._E[j][o])*model._start[k, q] + (model._pt[j][o]+model._E[j][o]-model._E[k][q])*model._start[j, o] >= model._pt[k][q]*model._pt[j][o]+model._E[k][q]*model._pt[j][o]+model._E[j][o]*model._pt[k][q]) for j in range(
        model._n_jobs) for o in range(model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if ((model._ma[j][o] == model._ma[k][q]) and (model._E[j][o] < model._E[k][q] + model._pt[k][q]) and (model._E[k][q] < model._E[j][o] + model._pt[j][o])))

    return model


def semi_activate(model):

    active_sol = [[model._start[j, o].X for o in range(
        model._job_lengths[j])] for j in range(model._n_jobs)]
    starts = [model._start[j, o].X for j in range(model._n_jobs) for o in range(
        model._job_lengths[j])]

    ops_id = [[j, o] for j in range(model._n_jobs)
              for o in range(model._job_lengths[j])]

    order = np.argsort(starts)
    machine_free = np.zeros(model._n_machines)

    # Loop through operations by increasing start time
    for id in order:
        j, o = ops_id[id]
        if o == 0:
            # Try shifting (min for compatibility with partial schedules)
            active_sol[j][o] = min(
                active_sol[j][o], machine_free[model._ma[j][o]])
        else:
            # Try shifting (idem for min)
            active_sol[j][o] = min(active_sol[j][o], max(machine_free[model._ma[j][o]],
                                   active_sol[j][o-1] + model._pt[j][o-1] + model._wt[j][o-1]))

        # Update when machine becomes free (idem for max)
        if not hasattr(model, "_scheduled") or model._scheduled[j, o].X >= .9:
            machine_free[model._ma[j][o]] = max(
                machine_free[model._ma[j][o]], active_sol[j][o] + model._pt[j][o])

    return active_sol


def callback_half_cuts(model, where):

    if where == GRB.Callback.MIPNODE:

        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:

            for j in range(model._n_jobs):
                for o in range(model._job_lengths[j]):

                    sum_term = 0

                    ma_list = model._ma_dict[model._ma[j][o]]
                    index = 0
                    current_op = ma_list[index]

                    while current_op != [j, o]:

                        if current_op[0] < j:
                            sum_term += model._pt[current_op[0]][current_op[1]] * \
                                model.cbGetNodeRel(model._before[current_op[0],
                                                                 current_op[1], j, o])
                        elif current_op[0] > j:
                            sum_term += model._pt[current_op[0]][current_op[1]] * \
                                (1-model.cbGetNodeRel(model._before[j, o, current_op[0],
                                                                    current_op[1]]))

                        index += 1
                        current_op = ma_list[index]

                    best_index = index
                    best_value = model._E[j][o] + sum_term

                    index += 1
                    while index < len(ma_list):
                        current_op = ma_list[index]

                        if current_op[0] < j:
                            sum_term += model._pt[current_op[0]][current_op[1]] * \
                                model.cbGetNodeRel(model._before[current_op[0],
                                                                 current_op[1], j, o])
                        elif current_op[0] > j:
                            sum_term += model._pt[current_op[0]][current_op[1]] * \
                                (1-model.cbGetNodeRel(model._before[j, o, current_op[0],
                                                                    current_op[1]]))

                        if model._E[current_op[0]][current_op[1]] + sum_term >= best_value:
                            best_value = model._E[current_op[0]
                                                  ][current_op[1]] + sum_term
                            best_index = index

                        index += 1

                    if model.cbGetNodeRel(model._start[j, o]) < best_value:
                        best_E = model._E[ma_list[best_index]
                                          [0]][ma_list[best_index][1]]
                        rhs = gp.LinExpr(best_E)
                        for index in range(best_index+1):
                            k, q = ma_list[index]
                            if k < j:
                                rhs.add(
                                    model._before[k, q, j, o], model._pt[k][q])
                            elif k > j:
                                rhs.add(
                                    (1-model._before[j, o, k, q]), model._pt[k][q])

                        model.cbCut(model._start[j, o] >= rhs)
                        return
    return


def update_for_callback(model):

    model.Params.PreCrush = 1

    model._E = []
    for j in range(model._n_jobs):
        tmp = [0]
        for o in range(1, model._job_lengths[j]):
            tmp.append(tmp[o-1]+model._pt[j][o-1])
        model._E.append(tmp)
    model._ma_dict = {}
    for mach in range(model._n_machines):
        tmp_list = []
        tmp_E = []
        for j in range(model._n_jobs):
            for o in range(model._job_lengths[j]):
                if model._ma[j][o] == mach:
                    tmp_list.append([j, o])
                    tmp_E.append(model._E[j][o])

        # Add operations in decreasing order of earliest possible start
        model._ma_dict[mach] = [x for _, x in sorted(
            zip(tmp_E, tmp_list), reverse=True)]

    return model


def print_line():
    print("------------------------------------------------------")


def compute_heads_tails(model):

    model._head = []
    model._tail = []

    for j in range(model._n_jobs):
        tmp_head = [0]
        tmp_tail = [sum(model._pt[j])+sum(model._wt[j])]
        for o in range(model._job_lengths[j]-1):
            tmp_head.append(tmp_head[o]+model._pt[j][o]+model._wt[j][o])
            tmp_tail.append(tmp_tail[o]-model._pt[j][o]-model._wt[j][o])

        model._head.append(tmp_head)
        model._tail.append(tmp_tail)

    return model


def reset_indicator_constraints(model):

    model.remove(model._machine1)
    model.remove(model._machine2)

    model._machine1 = model.addConstrs((((model._before[j, o, k, q] == 1) >> (model._start[k, q] >= model._start[j, o] + model._pt[j][o])) for j in range(model._n_jobs) for o in range(
        model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine1")
    model._machine2 = model.addConstrs((((model._before[j, o, k, q] == 0) >> (model._start[j, o] >= model._start[k, q] + model._pt[k][q])) for j in range(model._n_jobs) for o in range(
        model._job_lengths[j]) for k in range(j+1, model._n_jobs) for q in range(model._job_lengths[k]) if (model._ma[j][o] == model._ma[k][q])), name="machine2")

    return model
