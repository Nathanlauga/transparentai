import unittest
import sys
import os
import numpy as np
import pandas as pd
import datetime

import transparentai.utils as utils


class TestUtils(unittest.TestCase):

    def test_is_array_like(self):
        self.assertFalse(utils.is_array_like({'test':0}))
        self.assertFalse(utils.is_array_like(0))
        self.assertFalse(utils.is_array_like('sqdfd'))
        self.assertTrue(utils.is_array_like([1,2,3]))
        self.assertTrue(utils.is_array_like(np.array([1,2,3])))
        self.assertTrue(utils.is_array_like(pd.Series([1,2,3])))
        self.assertTrue(utils.is_array_like(pd.DataFrame([1,2,3])))
        self.assertFalse(utils.is_array_like([[1,2],[2,3],[3,4]]))
        self.assertFalse(utils.is_array_like(np.array([[1,2],[2,3],[3,4]])))
        self.assertFalse(utils.is_array_like(pd.Series([[1,2],[2,3],[3,4]])))
        self.assertFalse(utils.is_array_like(pd.DataFrame([[1,2],[2,3],[3,4]])))

    def test_find_dtype(self):
        self.assertRaises(TypeError, utils.find_dtype)
        self.assertEqual(utils.find_dtype([1,2]),'number')
        self.assertEqual(utils.find_dtype(['1','2']),'number')
        self.assertEqual(utils.find_dtype([datetime.date(1958,5,12),datetime.date(1980,12,12)]),'datetime')
        self.assertEqual(utils.find_dtype(['blabla','2']),'object')
        self.assertEqual(utils.find_dtype(pd.DataFrame([1,2])),'number')
        self.assertEqual(utils.find_dtype(pd.Series(['1','2'])),'number')
        self.assertEqual(utils.find_dtype(pd.Series([datetime.date(1958,5,12),datetime.date(1980,12,12)])),'datetime')
        self.assertEqual(utils.find_dtype(pd.DataFrame(['blabla','2'])),'object')


if __name__ == '__main__':
    unittest.main()
