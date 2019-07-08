import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

from models import LSTM_BRNN, Residual_CNN
from reader import Dataset, TRAIN_MODE, DEV_MODE, TEST_MODE

SUPPORTED_INPUT_TYPES = ['ONEHOT', 'BLOSUM', 'PHYS', 'PSIPRED', 'SOLVPRED', 'PSSM', 'TRUE_SS']
SUPPORTED_OUTPUT_TYPES = ['PHI', 'PSI']
SUPPORTED_MODELS = ['lstm_brnn', 'residual_cnn']

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--input_types', nargs='+', default=['ONEHOT'], help='Input types')
parser.add_argument('--output_types', nargs='+', default=['PHI', 'PSI'], help='Output types')
parser.add_argument('--model_type', type=str, default='lstm_brnn', help='Model type')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--iterations', type=int, default=10000, help='Number training iterations')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizers')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')

# Helper Functions
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_l2loss_from_predictions(network_outputs, true_outputs, masks):
    error = true_outputs - network_outputs
    masked_error = error * masks

    l2_loss = torch.sum(masked_error ** 2) / 2

    return l2_loss

def _initialize_model(model_type, input_dim, output_dim):
    if model_type == 'residual_cnn':
        return Residual_CNN(input_dim=input_dim, output_dim=output_dim)

    if model_type == 'lstm_brnn':
        return LSTM_BRNN(input_dim=input_dim, hidden_dim=256, num_layers=2,
                         fc_dims=[1024, 512], output_dim=output_dim)

def evaluate(model, dataset, mode, nb_outputs, cuda=None, device_id=None):
    if mode == TRAIN_MODE: nb_examples = dataset.train.size
    if mode == DEV_MODE: nb_examples = dataset.dev.size
    if mode == TEST_MODE: nb_examples = dataset.test.size

    maes = [[] for i in range(nb_outputs)]
    for _ in range(nb_examples):
        ids, lengths, inputs, _, _, angles = dataset.next_batch(1, mode)
        if cuda: inputs = inputs.cuda(device_id)
        outputs = model(inputs, lengths).squeeze().cpu().data.numpy()
        outputs = outputs * 2.0 - 1.0

        for i in range(lengths[0]):
            for j in range(nb_outputs):
                predicted_sin, predicted_cos = outputs[i, j*2], outputs[i, j*2+1] + 1e-8
                predicted_angle = np.rad2deg(np.arctan2(predicted_sin,predicted_cos))
                correct_angle = angles[0][i][j]
                if correct_angle != 360:
                    d = np.absolute(predicted_angle - correct_angle)
                    maes[j].append(min(360-d, d))
    return [np.average(_maes) for _maes in maes]

# Main Function
def main():
    # Arguments Parsing
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true' and torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    device_id = args.device_id

    input_types = args.input_types
    output_types = args.output_types
    model_type = args.model_type

    # Checks
    for input_type in input_types:
        assert(input_type in SUPPORTED_INPUT_TYPES)
    for output_type in output_types:
        assert(output_type in SUPPORTED_OUTPUT_TYPES)
    assert(model_type in SUPPORTED_MODELS)

    save_path = 'model/{}'.format(model_type)
    for input_type in input_types: save_path = save_path + '_' + input_type.lower()
    iterations = args.iterations
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_interval = args.log_interval

    # Load dataset
    dataset = Dataset(input_types, output_types)
    print('Loaded dataset')

    # Load model
    model = _initialize_model(model_type, dataset.input_dim, dataset.output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if cuda:
        model.cuda(device_id)

    # Start Training
    best_dev_avg_mae = None
    for itx in range(iterations):
        model.train()
        model.zero_grad()

        _, lengths, inputs, outputs, masks, _ = dataset.next_batch(batch_size, TRAIN_MODE)
        if cuda:
            inputs = inputs.cuda(device_id)
            outputs = outputs.cuda(device_id)
            masks = masks.cuda(device_id)

        network_outputs = model(inputs, lengths)
        l2_loss = get_l2loss_from_predictions(network_outputs, outputs, masks)
        l2_loss.backward()
        optimizer.step()

        if itx % args.log_interval == 0:
            model.eval()
            dev_maes = evaluate(model, dataset, DEV_MODE, len(output_types), cuda, device_id)
            test_maes = evaluate(model, dataset, TEST_MODE, len(output_types), cuda, device_id)

            print("---------------------")
            print("iters:", itx)
            print("Dev MAEs:", dev_maes)
            print("Test MAEs:", test_maes)

            dev_avg_mae = np.mean(dev_maes)

            if best_dev_avg_mae == None or best_dev_avg_mae > dev_avg_mae:
                # Save the model
                create_dir_if_not_exists('model')
                torch.save(model, save_path)
                print('Saved the model')
                best_dev_avg_mae = dev_avg_mae

if __name__=="__main__":
    main()
