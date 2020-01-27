import unittest
import sys
import os
import pandas as pd


class TestBiasMetrics(unittest.TestCase):

    data = pd.DataFrame([
        ['Male',25,1],
        ['Male',50,0],
        ['Male',39,0],
        ['Male',68,1],
        ['Male',45,1],
        ['Male',38,1],
        ['Female',42,1],
        ['Female',27,1],
        ['Female',74,0],
        ['Female',22,0]
    ], columns=['gender','age','target'])
    preds = [0,1,0,1,0,1,0,1,0,1]

    def test_dataset_metrics(self):
        # TODO
        self.assertEqual(1,1)


if __name__ == '__main__':
    unittest.main()
