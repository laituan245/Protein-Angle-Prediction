import json
import random
import numpy as np

from utils import read_json, AugmentedList
from utils import PHYS_DICT, AA_DICT, BLOSUM_DICT, SS_DICT
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

TRAIN_MODE = 0
DEV_MODE = 1
TEST_MODE = 2

# Dataset Class
class Dataset:
    def __init__(self, input_types=['ONEHOT'], output_types=['PHI', 'PSI']):
        self.input_types = input_types
        self.output_types = output_types

        # Calculate input_dim
        input_dim = 0
        for input_type in input_types:
            if input_type == 'ONEHOT': input_dim += 20
            if input_type == 'BLOSUM': input_dim += 20
            if input_type == 'PHYS': input_dim += 7
            if input_type == 'PSIPRED': input_dim += 3
            if input_type == 'SOLVPRED': input_dim += 1
            if input_type == 'PSSM': input_dim += 20
            if input_type == 'TRUE_SS': input_dim += 8
        self.input_dim = input_dim

        # Calculate output_dim
        self.output_dim = 2 * len(output_types)

        train_data = read_json('data/train.json')
        dev_data = read_json('data/dev.json')
        test_data = read_json('data/test.json')

        self.train = AugmentedList(list(train_data.items()), shuffle_between_epoch=True)
        self.dev = AugmentedList(list(dev_data.items()))
        self.test = AugmentedList(list(test_data.items()))

        # Read features file
        pssm_1 = read_json('data/features/pssm_1.json')
        pssm_2 = read_json('data/features/pssm_2.json')
        pssm_3 = read_json('data/features/pssm_3.json')
        pssm_4 = read_json('data/features/pssm_4.json')
        pssm_5 = read_json('data/features/pssm_5.json')
        self.pssm = {}
        for d in [pssm_1, pssm_2, pssm_3, pssm_4, pssm_5]:
            for k, v in d.items():
                self.pssm[k] = v

        self.psipred = read_json('data/features/psipred.json')
        self.solvpred = read_json('data/features/solvpred.json')

    def next_batch(self, batch_size, mode=TRAIN_MODE):
        if mode == TRAIN_MODE: examples = self.train.next_items(batch_size)
        elif mode == DEV_MODE: examples = self.dev.next_items(batch_size)
        elif mode == TEST_MODE: examples = self.test.next_items(batch_size)

        lengths = [len(example[1]['seq']) for example in examples]
        max_len = max(lengths)
        # Create output masks
        masks = []
        for length in lengths:
            mask = [[1] * self.output_dim] * length + [[0] * self.output_dim] * (max_len - length)
            masks.append(mask)
        masks = Variable(FloatTensor(masks))

        row = 0
        ids, inputs, outputs, angles = [], [], [], []
        for example in examples:
            _id = example[0]
            ids.append(_id)
            seq = example[1]['seq']
            seq_length = len(seq)
            phi, psi, true_ss = example[1]['phi'], example[1]['psi'], example[1]['true_ss']
            psipreds = self.psipred.get(_id, [[0, 0, 0]] * len(seq))
            solvpreds = self.solvpred.get(_id, [0.0] * len(seq))
            pssms = self.pssm.get(_id, [[0] * 20] * len(seq))

            # Construct _inputs
            _inputs = []
            for i in range(seq_length):
                _input = []
                for input_type in self.input_types:
                    if input_type == 'ONEHOT': _input += AA_DICT[seq[i]]
                    if input_type == 'BLOSUM': _input += BLOSUM_DICT[seq[i]]
                    if input_type == 'PHYS': _input += PHYS_DICT[seq[i]]
                    if input_type == 'PSIPRED': _input += psipreds[i]
                    if input_type == 'SOLVPRED': _input += [solvpreds[i]]
                    if input_type == 'PSSM': _input += pssms[i]
                    if input_type == 'TRUE_SS': _input += SS_DICT[true_ss[i]]
                _inputs.append(_input)
            _inputs += [[0] * self.input_dim] * (max_len - len(_inputs))
            inputs.append(_inputs)

            # Construct _angles and _outputs
            _angles, _outputs = [], []
            for i in range(seq_length):
                _angle, _output = [], []
                j = 0
                for output_type in self.output_types:
                    if output_type == 'PHI': angle = np.float(phi[i])
                    if output_type == 'PSI': angle = np.float(psi[i])
                    _angle.append(angle)
                    _output += [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
                    if angle == 360.0:
                        masks[row, i, j * 2] = 0
                        masks[row, i, j * 2 + 1] = 0
                    j += 1
                _outputs.append(_output)
                _angles.append(_angle)
            _outputs += [[0] * self.output_dim] * (max_len - len(_outputs))
            angles.append(_angles)
            outputs.append(_outputs)

            row += 1

        # Sorted in decreasing order of length
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if lengths[i] < lengths[j]:
                    ids[i], ids[j] = ids[j], ids[i]
                    lengths[i], lengths[j] = lengths[j], lengths[i]
                    inputs[i], inputs[j] = inputs[j], inputs[i]
                    outputs[i], outputs[j] = outputs[j], outputs[i]
                    x_row = masks[i,:, :].clone()
                    masks[i,:, :] = masks[j,:, :]
                    masks[j,:, :] = x_row

        # Convert to FloatTensors
        inputs = Variable(FloatTensor(inputs))
        outputs = Variable(FloatTensor(outputs))

        # scale outputs to 0 to 1 range. sigmoid activation.
        outputs = (outputs + 1.0) / 2.0

        return ids, lengths, inputs, outputs, masks, angles
