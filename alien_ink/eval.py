#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random


def random_class_predictions(count=10, choices=(0, 1)):
    """Return a list of random predictions from a choice list

    """
    random.seed()
    predictions = [random.choice(choices) for i in range(count)]
    return predictions



