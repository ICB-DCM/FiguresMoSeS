import pandas as pd
import modelSelector
from modelSelector.model import ModelBase

df = pd.read_csv('timings.tsv', sep='\t')
df.set_index('model_id', inplace=True)
df.loc['M_1111111111111111']


class FAMOS_Model(ModelBase):

    def fit(self):
        self.nll = df.loc[self.id.upper()]['NLLH']


super_model = FAMOS_Model('M_1111111111111111')
mini_model = FAMOS_Model('M_0000000000000000')
criterion = modelSelector.AIC()

result = modelSelector.backward_search(super_model, criterion=criterion, exclude_before_fitting=True)

result_backward_full = modelSelector.backward_search(super_model, criterion=criterion, exclude_before_fitting=False)

result_forward = modelSelector.forward_search(mini_model, criterion=criterion)

# result_brute_force = modelSelector.brute_force_search(super_model, criterion=criterion)
result_brute_force = df['AIC'].to_dict()

result_exhaustive = modelSelector.efficient_exhaustive_search(super_model, criterion=criterion)

result_exhaustive_from_forward = \
    modelSelector.backward_search(super_model, criterion=criterion, result=result_forward)


total_time = 0
for model_id in result_forward:
    if result_forward[model_id]:
        total_time += df.loc[model_id]['total_time']

print(total_time)


