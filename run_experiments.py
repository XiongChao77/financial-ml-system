import os,sys,time
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation
from model import train
from trade.bt import simulation

def main():
    # preparation.main()
    # train.main()
    simulation.main()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f": run_time: {(time.time() - start_time):.4f} s")