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

    def test_get_locale_language(self):
        self.assertEqual(utils.get_locale_language('en-US,en;q=0.5'),'en')
        self.assertEqual(utils.get_locale_language('fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5'),'fr')
        self.assertEqual(utils.get_locale_language('de-CH'),'en')
        

if __name__ == '__main__':
    unittest.main()