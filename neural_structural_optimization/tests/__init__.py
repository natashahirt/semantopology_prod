"""Test suite for neural structural optimization.

Deliberately does not import the test modules. Eager imports here mean a single
missing optional dependency (for example apache_beam, used only by
test_pipeline) aborts collection for every other test module as well.
"""
