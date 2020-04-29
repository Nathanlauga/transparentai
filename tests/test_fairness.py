import unittest
import pandas as pd

import transparentai.fairness as fairness


class TestFairness(unittest.TestCase):

    data = pd.DataFrame([['Male', 90],
                         ['Female', 70],
                         ['Male', 52],
                         ['Male', 10],
                         ['Female', 25]], columns=['gender', 'age'])

    def test_create_privilieged_df(self):
        privileged_group = {
            # privileged group is man for gender attribute
            'gender': ['Male'],
            # privileged group aged between 30 and 55 years old
            'age': lambda x: (x > 30) & (x < 55)
        }

        privileged_df = fairness.create_privilieged_df(self.data,
                                                       privileged_group)

        self.assertEqual(type(privileged_df), pd.DataFrame)
        self.assertEqual(privileged_df.iloc[0, 0], 1)
        self.assertEqual(privileged_df.iloc[0, 1], 0)

        privileged_group['gender'] = ['None']

        privileged_df = fairness.create_privilieged_df(self.data,
                                                       privileged_group)

        self.assertEqual(privileged_df['gender'].sum(), 0)


if __name__ == '__main__':
    unittest.main()
