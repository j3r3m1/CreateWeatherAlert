# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:43:51 2020

@author: Jérémy Bernard
"""
import unittest

class TestAnalyseData(unittest.TestCase):

    def test_filterTimeAndAverage(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
