import os,sys,time
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation
from model import train
from trade.bt import simulation

def main():
    begin_time = time.time()
    preparation_start_time = begin_time
    # preparation.main()
    train_start_time = time.time()
    train.main()
    simulation_start_time = time.time()
    simulation.main()
    end_time = time.time()
    print(f": preparation run_time: {(train_start_time - preparation_start_time):.4f} s | train run_time: {(simulation_start_time - train_start_time):.4f} s "
          f"| simulation run_time: {(end_time - simulation_start_time):.4f} s |total : {(end_time - begin_time):.4f} s")

if __name__ == "__main__":
    main()