import unittest
import pandas as pd
import numpy as np

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

    def test_fairness_metrics(self):
        y_pred = np.array([1, 0, 0, 1, 1])
        y_true = np.array([1, 0, 0, 0, 1])
        privileged_group = {
            # privileged group is man for gender attribute
            'gender': ['Male']
        }
        # TP = 2, TP_F = 1, TP_M = 1
        # FP = 1, FP_F = 0, FP_M = 1
        # P = 3, P_F = 2, P_M = 1
        # N = 2, N_F = 0, N_M = 2
        # TPR = 0.66, TPR_F = 0.5, TPR_M = 1

        male_pred   = y_pred[self.data['gender'] == 'Male']
        female_pred = y_pred[self.data['gender'] == 'Female']
        male_true   = y_true[self.data['gender'] == 'Male']
        female_true = y_true[self.data['gender'] == 'Female']

        male_pos_pred   = male_pred.sum()
        female_pos_pred = female_pred.sum()
        male_neg_pred   = (male_pred == 0).sum()
        female_neg_pred = (female_pred == 0).sum()

        p_male_pos   = male_pos_pred / len(male_pred)
        p_female_pos = female_pos_pred / len(female_pred)

        tp_male  = ((male_pred == 1) & (male_true == 1)).sum()
        fp_male  = ((male_pred == 1) & (male_true == 0)).sum()
        tpr_male = tp_male / male_true.sum()
        fpr_male = fp_male / (male_true == 0).sum()

        tp_female  = ((female_pred == 1) & (female_true == 1)).sum()
        fp_female  = ((female_pred == 1) & (female_true == 0)).sum()
        tpr_female = tp_female / female_true.sum()
        fpr_female = fp_female / (female_true == 0).sum()

        b = y_pred - y_true + 1

        statistical_parity_difference = p_female_pos - p_male_pos
        disparate_impact              = p_female_pos / p_male_pos
        equal_opportunity_difference  = tpr_female - tpr_male
        average_odds_difference       = (1/2) * ((fpr_female-fpr_male) + (tpr_female-tpr_male))
        theil_index                   = np.mean(np.log((b / np.mean(b))**b) / np.mean(b))
        

        res = fairness.compute_fairness_metrics(y_true,
                                                y_pred,
                                                self.data,
                                                privileged_group)

        res = res['gender']

        self.assertEqual(res['statistical_parity_difference'], 
                            statistical_parity_difference)
        self.assertEqual(res['disparate_impact'], 
                            disparate_impact)
        self.assertEqual(res['equal_opportunity_difference'], 
                            equal_opportunity_difference)
        self.assertEqual(res['average_odds_difference'], 
                            average_odds_difference)
        self.assertEqual(res['theil_index'], 
                            theil_index)

if __name__ == '__main__':
    unittest.main()
