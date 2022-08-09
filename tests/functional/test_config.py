#!/usr/bin/env python


from importlib import reload
import os
import unittest

from alien_ink import config


class TestConfig(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_config_defaults(self):
        reload(config)
        self.assertTrue(config._config_file_ink, "alien-ink.json")


if __name__ == "__main__":

    unittest.main()
