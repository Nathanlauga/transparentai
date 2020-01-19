import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')+'/transparentai/app'))
import utils

class TestData(unittest.TestCase):

    def test_is_path_format_correct(self):
        self.assertFalse(utils.is_path_format_correct('path/without'))
        self.assertTrue(utils.is_path_format_correct('path/with/'))

    def test_remove_empty_from_dict(self):
        test_dict = {'1':'ok', '2':'', '3':None}
        self.assertEqual(utils.remove_empty_from_dict(test_dict),{'1':'ok'})
        self.assertTrue('2' not in utils.remove_empty_from_dict(test_dict))
        self.assertFalse('3' in utils.remove_empty_from_dict(test_dict))

if __name__ == '__main__':
    unittest.main()