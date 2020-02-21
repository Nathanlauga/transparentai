import unittest
import sys
import os
import pandas as pd

import transparentai.utils as utils


class TestUtils(unittest.TestCase):

    data = pd.DataFrame([['1','2','3'],['1','2','3'],['2','2','2']], columns=['1','2','3'])

    def test_remove_var_with_one_value(self):
        test = utils.remove_var_with_one_value(self.data)

        self.assertEqual(test.shape[1],2)

    def test_encode_categorical_vars(self):
        test, _ = utils.encode_categorical_vars(self.data)
        
        self.assertIn(test.columns[0], test.select_dtypes('number'))

if __name__ == '__main__':
    unittest.main()
