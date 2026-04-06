import configparser


def load_config():
    import constants
    config = configparser.ConfigParser()
    # Load the configuration file
    config.read(constants.root+'/settings/config.ini')
    # Load configs
    for section in config.sections():
        for options in config.options(section):
            if options == 'root':
                constants.root = config.get(section, options)
            elif options == 'algo':
                constants.algo = config.get(section, options)
            elif options == 'workload':
                constants.workload = config.get(section, options)
            elif options == 'beta':
                constants.beta = float(config.get(section, options))
            elif options == 'iteration':
                constants.iteration = int(config.get(section, options))
            elif options == 'fixed_episodic_reward':
                constants.fixed_episodic_reward = int(config.get(section, options))
            elif options == 'epsilon':
                constants.epsilon = float(config.get(section, options))
            elif options == 'learning_rate':
                constants.learning_rate = float(config.get(section, options))
            elif options == 'gamma':
                constants.gamma = float(config.get(section, options))
            elif options == 'placement_penalty':
                constants.placement_penalty = int(config.get(section, options))
            elif options == 'pp_apply':
                constants.pp_apply = config.get(section, options)
            elif options == 'use_dirichlet_weights':
                constants.use_dirichlet_weights = config.getboolean(section, options)
            elif options == 'enable_stochastic_pricing':
                constants.enable_stochastic_pricing = config.getboolean(section, options)
            elif options == 'price_variance':
                constants.price_variance = float(config.get(section, options))
            elif options == 'job_batch_size':
                constants.job_batch_size = max(1, int(config.get(section, options)))
            elif options == 'use_recurrent_policy':
                constants.use_recurrent_policy = config.getboolean(section, options)
            elif options == 'trace_type':
                constants.trace_type = config.get(section, options)
            # ── Phase-2 flags ──────────────────────────────────────────────
            elif options == 'w_mode':
                constants.w_mode = config.get(section, options).strip()
            elif options == 'pricing_mode':
                constants.pricing_mode = config.get(section, options).strip()
            elif options == 'batch_b':
                constants.batch_B = max(1, int(config.get(section, options)))
            else:
                print('Invalid Option found {}'.format(options))
