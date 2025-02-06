import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import shutil
import itertools
import pandas as pd
import numpy as np

import petab
import amici.petab_simulate


minimize_options = {
    "n_starts": 1000,
    "optimizer": optimize.FidesOptimizer(),
    "engine": pypesto.engine.MultiProcessEngine()
}

petab_path = "m_true/example_modelSelection.yaml"

importer = pypesto.petab.PetabImporter.from_yaml(petab_path, simulator_type="amici")
problem = importer.create_problem() 

# set options
n_starts = 1000
# engine = pypesto.engine.MultiProcessEngine()
engine = None
optimizer = optimize.FidesOptimizer()

result = optimize.minimize(
    problem=problem, optimizer=optimizer, n_starts=n_starts, engine=engine
)




"""
Strategy
# generate synthetic data
# generate all the models
# fit them all
"""

# define the model Setup
ts = [0, 1, 5, 10, 30, 60]
# y_idxs = [1, 2] # which of the observables to keep, should only be 1 or 2
y_idxs = [1, 2] # [1, 2]

"""
Create the synthtic data
"""
print(f"Copy & modify files")

# Prep: change the OBSERVABLE + PARAMETER table
print("Update OBSERVABLE + PARAMETER table")

obs_table = pd.read_csv(f"m_true/observables_example_modelSelection.tsv", sep="\t")
# obs_table.set_index('observableId', inplace=True)

param_table = pd.read_csv(f"m_true/parameters_example_modelSelection.tsv", sep="\t")
# param_table.set_index('parameterId', inplace=True)


for y_idx in [1, 2]:
    if y_idx not in y_idxs:
        rows_to_drop = obs_table[obs_table['observableId'] == f"obs_x{y_idx}"].index
        obs_table.drop(rows_to_drop, inplace=True)

        rows_to_drop = param_table[param_table['parameterId'] == f"sigma_x{y_idx}"].index
        param_table.drop(rows_to_drop, inplace=True)

obs_table.to_csv(f"m_true/observables_example_modelSelection.tsv", sep="\t", index=False)
param_table.to_csv(f"m_true/parameters_example_modelSelection.tsv", sep="\t", index=False)

# Prep: create the (empty) measurment table:
print("Prepare the measurement table")

# msmt_table = pd.DataFrame(columns=["observableId", "simulationConditionId", "measurement", "time", "noiseParameters"])

msmt_list = []
for y_idx in y_idxs:
    for t in ts:
        msmt_list.append({"observableId": f"obs_x{y_idx}",
                          "simulationConditionId": "model1_data1",
                          "measurement": 0,
                          "time": t,
                          "noiseParameters": f"sigma_x{y_idx}"})

msmt_table = pd.DataFrame(msmt_list)
msmt_table.to_csv(f"m_true/measurementData_example_modelSelection_synthetic.tsv", sep="\t", index=False)

#  Prep: generate synthtic data for the model
print("Generate synthetic data")

# load the PEtab model of the ground truth:
petab_problem_synthetic = petab.Problem.from_yaml(
    f"m_true/example_modelSelection.yaml")

# simulate synthetic data
simulator = amici.petab_simulate.PetabSimulator(petab_problem_synthetic)
petab_problem_synthetic.measurement_df = simulator.simulate(noise=True, as_measurement=True)

# ground truth measurement file
petab_problem_synthetic.measurement_df.to_csv(
    f"m_true/measurementData_example_modelSelection_synthetic.tsv", sep='\t', index=False)

print("Done")

"""
Create all the different submodels and fit them
"""

results = dict()
for m_idxs in itertools.product([0, 1], repeat=3):
    # create the folder
    m_name = f"m_{m_idxs[0]}{m_idxs[1]}{m_idxs[2]}"
    if m_name == "m_000":
        pass
        # continue
    os.makedirs(f"{m_name}", exist_ok=True)

    print("Fitting model: ", m_name)
    # copy the PEtab files
    for file_name in ["experimentalCondition_example_modelSelection.tsv", 
                      "measurementData_example_modelSelection_synthetic.tsv", 
                      "observables_example_modelSelection.tsv", 
                      "parameters_example_modelSelection.tsv",
                      "model_example_modelSelection.xml", 
                      "example_modelSelection.yaml"]:

        shutil.copy(f"m_true/{file_name}", 
                    f"{m_name}/{file_name}")

    # update the parameter table
    param_table = pd.read_csv(f"{m_name}/parameters_example_modelSelection.tsv", sep="\t")
    param_table.set_index('parameterId')
    for i, m_idx in enumerate(m_idxs):
        if m_idx == 0:
            param_table.loc[param_table['parameterId'] == f'k{i+1}', 'estimate'] = 0
            param_table.loc[param_table['parameterId'] == f'k{i+1}', 'nominalValue'] = 0

    param_table.to_csv(f"{m_name}/parameters_example_modelSelection.tsv", sep="\t", index=False)
    
    # fit the model
    importer = pypesto.petab.PetabImporter.from_yaml(
        f'{m_name}/example_modelSelection.yaml', 
        simulator_type="amici", 
        model_name=f"{m_name}_v{num_experiment}"
        )
    problem = importer.create_problem(force_compile=True) 

    # set options
    n_starts = 100
    # engine = pypesto.engine.MultiProcessEngine()
    engine = None
    optimizer = optimize.FidesOptimizer()

    results[m_name] = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=n_starts, engine=engine
    )

# copy from the ground truth version to a new model w. the right active parameters

print('LLH:\n=================')
# print the different results
for key, val in results.items():
    print(key, val.optimize_result.fval[0])

# write the results to file
with open(f"results.tsv", "w") as f:
    for key, val in results.items():
        f.write(f"{key}\t{val.optimize_result.fval[0]}\n")

print('AIC:\n=================')
# print the different results
for key, val in results.items():
    n_params = key.count("1") + 2
    n_data = len(ts) * len(y_idxs)
    print(key)
    print("NLL", val.optimize_result.fval[0])
    print("AIC: ", 2*val.optimize_result.fval[0] + 2 * n_params)
    print("BIC: ", 2*val.optimize_result.fval[0] + n_params * np.log(n_data))
    print("AICC: ", 2*val.optimize_result.fval[0] + 2*key.count("1") + (2*n_params**2 + 2*n_params)/(n_data - n_params - 1))
    print("=====================================")



print(f"Model M{int(m_idxs[0])}{int(m_idxs[1])}{int(m_idxs[2])}: {result.optimize_result.fval[0]}")
