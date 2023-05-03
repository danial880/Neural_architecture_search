import utils
import logging
import numpy as np
from network_mix import NetworkMixArch


class ModelSearch():
    """
    Class for searching depth and width of the model, operations, and kernels.

    Args:
        args: A namespace object containing the command line arguments.
        class_labels: A list of class labels for the dataset.

    Attributes:
        CLASSES (int): The number of classes in the dataset.
        target_acc (float): The target accuracy of the model.
        min_width (int): The minimum width of the model.
        max_width (int): The maximum width of the model.
        width_resolution (int): The resolution of width search space.
        min_depth (int): The minimum depth of the model.
        max_depth (int): The maximum depth of the model.
        ch_drop_tolerance (float): The tolerance for channel dropping.
        target_acc_tolerance (float): The tolerance for the target accuracy.
        channels (int): The number of channels in the model.
        layers (int): The number of layers in the model.
        dataset (str): The name of the dataset being used for training.
        input_shape (list): The shape of the input data.
    """
    def __init__(self, config, class_labels):
        search_params = config['search_parameters']
        self.CLASSES = len(np.array(class_labels))
        self.target_acc = search_params['target_acc']
        self.min_width = search_params['min_width_channels']
        self.max_width = search_params['max_width_channels']
        self.width_resolution = search_params['width_resolution']
        self.min_depth = search_params['min_depth_layers']
        self.max_depth = search_params['max_depth_layers']
        self.ch_drop_tolerance = search_params['ch_drop_tolerance']
        self.target_acc_tolerance = search_params['target_acc_tolerance']
        # We start with max width but with min depth.
        self.channels = self.max_width
        self.layers = self.min_depth
        self.input_shape = config['hyperparameters']['input_shape']

    def intialize_model(self, arch_ops, arch_kernel):
        model = NetworkMixArch(self.channels, self.CLASSES, self.layers,
                               arch_ops, arch_kernel, self.input_shape)
        model = model.cuda()
        return model

    def log_model_details(self, model, arch_ops, arch_kernel):
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", self.layers,
                     self.channels)
        logging.info("Model self.layers %s Model Kernels %s", arch_ops,
                     arch_kernel)
        logging.info("Model Parameters = %fMB",
                     utils.count_parameters_in_MB(model))
        logging.info('Training Model...')

    def log_acc(self, train_acc, test_acc):
        logging.info("Best Training Accuracy %f Best Validation Accuracy %f",
                     train_acc, test_acc)

    def search_depth_and_width(self, class_labels, train_test, train):

        logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

        # Initialize
        curr_arch_ops = next_arch_ops = np.zeros((self.layers,), dtype=int)
        curr_arch_kernel = next_arch_kernel = 3 * np.ones((self.layers,),
                                                          dtype=int)
        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0
        logging.info('RUNNING DEPTH SEARCH FIRST...')
        model = self.intialize_model(curr_arch_ops, curr_arch_kernel)
        self.log_model_details(model, curr_arch_ops, curr_arch_kernel)
        curr_arch_train_acc, curr_arch_test_acc = train_test(model)
        self.log_acc(curr_arch_train_acc, curr_arch_test_acc)
        # Search depth
        diff_target_acc = self.target_acc - self.target_acc_tolerance
        while ((curr_arch_test_acc < (diff_target_acc)) and
               (self.layers != self.max_depth)):
            # The possibility exists if trained for too long.
            if (curr_arch_train_acc == 100):
                break
            else:
                # prepare next candidate architecture.
                self.layers += 1
                next_arch_ops = np.zeros((self.layers,), dtype=int)
                next_arch_kernel = 3 * np.ones((self.layers,), dtype=int)
                model = self.intialize_model(next_arch_ops, next_arch_kernel)
                utils.log_hash()
                logging.info('Moving to Next Candidate Architecture...')
                self.log_model_details(model, next_arch_ops, next_arch_kernel)
                next_arch_train_acc, next_arch_test_acc = train_test(model)
                self.log_acc(next_arch_train_acc, next_arch_test_acc)
                # As long as we get significant improvement by increasing depth
                if (next_arch_test_acc >= curr_arch_test_acc + 0.25):
                    # update current architecture.
                    curr_arch_ops = next_arch_ops
                    curr_arch_kernel = next_arch_kernel
                    curr_arch_train_acc = next_arch_train_acc
                    curr_arch_test_acc = next_arch_test_acc
                    # But we still keep trying deeper candidates.
                elif (next_arch_test_acc >= curr_arch_test_acc - 0.15):
                    continue
                else:
                    break
        # During width search lenght of curr_arch_ops and curr_arch_kernel
        # shall not change but only channels.
        # discovered final number of self.layers
        f_layers = len(curr_arch_ops)
        # discovered final number of channels
        f_channels = self.max_width
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('END OF DEPTH SEARCH...')
        best_arch_test_acc = curr_arch_test_acc
        utils.log_hash()
        logging.info('RUNNING WIDTH SEARCH NOW...')
        while (self.channels > self.min_width):
            # prepare next candidate architecture.
            self.channels = self.channels - self.width_resolution
            # Although these do not change.
            model = self.intialize_model(curr_arch_ops, curr_arch_kernel)
            logging.info('Moving to Next Candidate Architecture...')
            self.log_model_details(model, curr_arch_ops, curr_arch_kernel)
            # train and test candidate architecture.
            next_arch_train_acc, next_arch_test_acc = train_test(model)
            self.log_acc(next_arch_train_acc, next_arch_test_acc)
            diff_best_acc = best_arch_test_acc - self.ch_drop_tolerance
            if (next_arch_test_acc >= (diff_best_acc)):
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
                f_channels = self.channels
            else:
                break
        logging.info('Discovered Final Width %s', f_channels)
        logging.info('END OF WIDTH SEARCH...')
        utils.log_hash()
        return curr_arch_ops, curr_arch_kernel, f_channels, f_layers, \
            curr_arch_train_acc, curr_arch_test_acc

    def search_operations(self, class_labels, train_test, train,
                          model_info):

        logging.info('RUNNING OPERATION SEARCH...')

        curr_arch_ops = model_info['curr_arch_ops']
        curr_arch_kernel = model_info['curr_arch_kernel']
        curr_arch_train_acc = model_info['curr_arch_train_acc']
        curr_arch_test_acc = model_info['curr_arch_test_acc']
        self.channels = model_info['f_channels']
        self.layers = model_info['f_layers']

        next_arch_ops = curr_arch_ops
        next_arch_kernel = curr_arch_kernel

        for i in range(self.layers):

            next_arch_ops[i] = 1
            model = self.intialize_model(next_arch_ops, next_arch_kernel)
            self.log_model_details(model, next_arch_ops, next_arch_kernel)
            next_arch_train_acc, next_arch_test_acc = train_test(model)
            self.log_acc(next_arch_train_acc, next_arch_test_acc)
            if next_arch_test_acc > curr_arch_test_acc:
                curr_arch_ops = next_arch_ops
                curr_arch_kernel = next_arch_kernel
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
            else:
                next_arch_ops[i] = 0

        logging.info('Discovered Final Operations %s', curr_arch_ops)
        logging.info('END OF OPERATION SEARCH...')
        utils.log_hash()
        return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, \
            curr_arch_test_acc

    def search_kernels(self, class_labels, train_test, train,
                       model_info):
        logging.info('RUNNING KERNEL SEARCH...')
        curr_arch_ops = model_info['curr_arch_ops']
        curr_arch_kernel = model_info['curr_arch_kernel']
        curr_arch_train_acc = model_info['curr_arch_train_acc']
        curr_arch_test_acc = model_info['curr_arch_test_acc']
        self.channels = model_info['f_channels']
        self.layers = model_info['f_layers']
        next_arch_ops = curr_arch_ops
        next_arch_kernel = curr_arch_kernel
        kernels = [5, 7]
        for i in range(self.layers):
            best_k = 3
            for k in kernels:
                next_arch_kernel[i] = k
                model = self.intialize_model(next_arch_ops, next_arch_kernel)
                self.log_model_details(model, next_arch_ops, next_arch_kernel)
                next_arch_train_acc, next_arch_test_acc = train_test(model)
                self.log_acc(next_arch_train_acc, next_arch_test_acc)
                # Bigger kernel comes at a cost therefore possibility of a
                # search hyper parameter exists.
                if (next_arch_test_acc > curr_arch_test_acc):
                    best_k = k
                    curr_arch_ops = next_arch_ops
                    curr_arch_kernel[i] = k
                    curr_arch_train_acc = next_arch_train_acc
                    curr_arch_test_acc = next_arch_test_acc
                else:
                    next_arch_kernel[i] = best_k
        logging.info('Discovered Final Kernels %s', curr_arch_kernel)
        logging.info('END OF KERNEL SEARCH...')
        utils.log_hash()
        return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, \
            curr_arch_test_acc

    def search_operations_and_kernels(self, model_info, class_labels,
                                      train_test, train):
        result_ops = self.search_operations(class_labels, train_test, train,
                                            model_info)
        curr_arch_ops, curr_arch_kernel = result_ops[:2]
        curr_arch_train_acc, curr_arch_test_acc = result_ops[2:]
        model_info['curr_arch_ops'] = curr_arch_ops
        model_info['curr_arch_kernel'] = curr_arch_kernel
        model_info['curr_arch_train_acc'] = curr_arch_train_acc
        model_info['curr_arch_test_acc'] = curr_arch_test_acc
        result_krnls = self.search_kernels(class_labels, train_test,
                                           train, model_info)
        curr_arch_ops, curr_arch_kernel = result_krnls[:2]
        curr_arch_train_acc, curr_arch_test_acc = result_krnls[2:]
        return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, \
            curr_arch_test_acc
