from __future__ import print_function

import numpy as np
import pytest
from ngraph.testing import executor

import ngraph as ng

pytestmark = [pytest.mark.transformer_dependent("module")]


def test_dimshuffle_op(transformer_factory):
    A = ng.make_axis().named('A')
    B = ng.make_axis().named('B')
    C = ng.make_axis().named('C')
    D = ng.make_axis().named('D')

    tests = [
        {
            'input_tensor': [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ],
                    [
                        [13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24],
                    ],
                ],
            ],
            'input_tensor_axes': (A, B, C, D),
            'output_tensor_axes': (B, D, A, C),
            'axes_lengths': {A: 1, B: 2, C: 3, D: 4},
            'expected_result': [
                [
                    [
                        [1, 5, 9]
                    ],
                    [
                        [2, 6, 10]
                    ],
                    [
                        [3, 7, 11]
                    ],
                    [
                        [4, 8, 12]
                    ],
                ],
                [
                    [
                        [13, 17, 21]
                    ],
                    [
                        [14, 18, 22]
                    ],
                    [
                        [15, 19, 23]
                    ],
                    [
                        [16, 20, 24]
                    ]
                ]
            ]
        },
        {
            'input_tensor': [
                [
                    [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ],
                    [
                        [13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24],
                    ],
                ],
                [
                    [
                        [25, 26, 27, 28],
                        [29, 30, 31, 32],
                        [33, 34, 35, 36],
                    ],
                    [
                        [37, 38, 39, 40],
                        [41, 42, 43, 44],
                        [45, 46, 47, 48],
                    ]
                ]
            ],
            'input_tensor_axes': (A, B, C, D),
            'output_tensor_axes': (B, D, A, C),
            'axes_lengths': {A: 2, B: 2, C: 3, D: 4},
            'expected_result': [
                [
                    [
                        [1, 5, 9],
                        [25, 29, 33],
                    ],
                    [
                        [2, 6, 10],
                        [26, 30, 34],
                    ],
                    [
                        [3, 7, 11],
                        [27, 31, 35],
                    ],
                    [
                        [4, 8, 12],
                        [28, 32, 36],
                    ]
                ],
                [
                    [
                        [13, 17, 21],
                        [37, 41, 45],
                    ],
                    [
                        [14, 18, 22],
                        [38, 42, 46],
                    ],
                    [
                        [15, 19, 23],
                        [39, 43, 47]
                    ],
                    [
                        [16, 20, 24],
                        [40, 44, 48],
                    ]
                ]
            ]
        },
    ]

    for test in tests:
        for axis, length in test['axes_lengths'].items():
            axis.length = length

        input_tensor = ng.placeholder(test['input_tensor_axes'])
        input_tensor_value = np.array(test['input_tensor'], dtype=np.float32)

        # This list of operations should add a dimshuffle operation to the graph.
        a = ng.negative(input_tensor)
        b = ng.axes_with_order(a, test['output_tensor_axes'])
        c = ng.negative(b)

        with executor(c, input_tensor) as ex:
            out = ex(input_tensor_value)
            ng.testing.assert_allclose(out, test['expected_result'])
