from __future__ import absolute_import, division, print_function

import numpy as np


class DataGenerator(object):

    def __init__(self):
        """Construct a DataGenerator."""
        pass

    def next_batch(self, batch_size, N, train_mode=True):
        """Return the next `batch_size` examples from this data set."""

        # A sequence of random numbers from [0, 1]
        reader_input_batch = []

        # Sorted sequence that we feed to encoder
        # In inference we feed an unordered sequence again
        decoder_input_batch = []

        # Ordered sequence where one hot vector encodes position in the input array
        writer_outputs_batch = []
        for _ in range(N):
            reader_input_batch.append(np.zeros([batch_size, 1]))
        for _ in range(N + 1):
            decoder_input_batch.append(np.zeros([batch_size, 1]))
            writer_outputs_batch.append(np.zeros([batch_size, N + 1]))

        for b in range(batch_size):
            shuffle = np.random.permutation(N)
            #print(shuffle)
            sequence = np.sort(np.random.random(N))
            #print(sequence)
            shuffled_sequence = sequence[shuffle]
            #print(shuffled_sequence)

            for i in range(N):
                reader_input_batch[i][b] = shuffled_sequence[i]
                if train_mode:
                    decoder_input_batch[i + 1][b] = sequence[i]
                else:
                    decoder_input_batch[i + 1][b] = shuffled_sequence[i]
                writer_outputs_batch[shuffle[i]][b, i + 1] = 1.0

            # Points to the stop symbol
            writer_outputs_batch[N][b, 0] = 1.0

        return reader_input_batch, decoder_input_batch, writer_outputs_batch
