import utilities
import constants
import cluster
import REINFORCE_tfagent
import DQN_tfagent
import workload
import R_DQN_tfagent
import QR_DQN_tfagent
import C51_tfagent



def main():
    utilities.load_config()
    workload.read_workload() 
    cluster.init_cluster()

    if constants.algo == 'reinforce':
        print("Running Reinforce Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        REINFORCE_tfagent.train_reinforce(num_iterations=constants.iteration)
    elif constants.algo == 'dqn':
        print("Running DQN Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        DQN_tfagent.train_dqn(num_iterations=constants.iteration)
    elif constants.algo == 'qr_dqn':
        print("Running QR DQN Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        QR_DQN_tfagent.train_qr_dqn(num_iterations=constants.iteration)
    elif constants.algo == 'rb_dqn':
        print("Running RainBow DQN Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        R_DQN_tfagent.train_rainbow_dqn(num_iterations=constants.iteration)
    elif constants.algo == 'c51':
        print("Running C51 DQN Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(constants.iteration, constants.workload, constants.beta))
        C51_tfagent.train_c51_dqn(num_iterations=constants.iteration)
    else:
        print('Please specify a valid algo option in the config.ini file\n')

if __name__ == '__main__':
    main()
