""" call to check on progress of checking modelspace -> change to corresponding dir """

from datetime import datetime
import math
from pathlib import Path


current_mo_size = 6
crit = 'AICC'

base_dir = Path(__file__).parent / crit
init_mo_dir = base_dir / f'initial_models_{current_mo_size}_parall.txt'
ex_mo_dir = base_dir / f'excluded_models_{current_mo_size}_parall.txt'


num_ex_mo = sum(1 for line in open(ex_mo_dir))
num_init_mo = sum(1 for line in open(init_mo_dir))

print(f'{round((num_init_mo+num_ex_mo)/math.comb(32, current_mo_size)*100, 4)}%'
      f'\t at {datetime.now()}')
