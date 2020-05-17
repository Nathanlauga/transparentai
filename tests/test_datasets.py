import unittest
import pandas as pd
import numpy as np

import transparentai.datasets as datasets


class TestFairness(unittest.TestCase):

    data = pd.DataFrame([['Male', 90, '1990-05-20'],
                         ['Female', 70, '1959-12-01'],
                         ['Male', 52, np.nan],
                         ['Male', 10, '1945-05-28'],
                         ['Female', 25, '2002-03-12']], columns=['gender', 'age', 'date'])
    data['date'] = pd.to_datetime(data['date'])

    def test_describe(self):
        self.assertRaises(TypeError, datasets.variable.describe)
        
        desc = datasets.variable.describe(self.data['age'])

        self.assertIn('quantile 75%', list(desc.keys()))
        self.assertEqual(desc['max'],90)
        self.assertEqual(desc['min'],10)
        self.assertEqual(desc['quantile 50%'],52)

        desc = datasets.variable.describe(self.data['gender'])

        self.assertIn('unique values', list(desc.keys()))
        self.assertEqual(desc['unique values'],2)
        self.assertEqual(desc['most common'],'Male')

        desc = datasets.variable.describe(self.data['date'])

        self.assertEqual(desc['valid values'],4)
        self.assertEqual(desc['missing values'],1)
        self.assertEqual(desc['max'],'2002-03-12')
        

if __name__ == '__main__':
    unittest.main()
