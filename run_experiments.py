import os,sys
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation
from model import train
from trade import simulation

preparation.main()
train.main()
simulation.main()