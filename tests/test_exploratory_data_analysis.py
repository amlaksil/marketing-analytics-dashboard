#!/usr/bin/python3
"""
Module for testing the ExploratoryDataAnalysis class.
This module includes tests for documentation, PEP8 compliance, and
various functionalities of the ExploratoryDataAnalysis class.
"""
import inspect
import unittest
from unittest.mock import patch, Mock, MagicMock
import matplotlib.pyplot as plt
import pandas as pd
import pycodestyle
from sqlalchemy import create_engine
from src.exploratory_data_analysis import ExploratoryDataAnalysis
import src
MODULE_DOC = src.exploratory_data_analysis.__doc__


class ExploratoryDataAnalysisDocs(unittest.TestCase):
    """
    Tests to check the documentation and style of ExploratoryDataAnalysis
    class.
    """
    def setUp(self):
        """Set up for docstring tests"""
        self.base_funcs = inspect.getmembers(
            ExploratoryDataAnalysis, inspect.isfunction)

    def test_pep8_conformance(self):
        """Test that src/exploratory_data_analysis.py conforms to PEP8."""
        for path in ['src/exploratory_data_analysis.py',
                     'tests/test_exploratory_data_analysis.py']:
            with self.subTest(path=path):
                errors = pycodestyle.Checker(path).check_all()
                self.assertEqual(errors, 0)

    def test_module_docstring(self):
        """Test for the existence of module docstring"""
        self.assertIsNot(MODULE_DOC, None,
                         "exploratory_data_analysis.py needs a docstring")
        self.assertTrue(len(MODULE_DOC) > 1,
                        "exploratory_data_analysis.py needs a docstring")

    def test_class_docstring(self):
        """Test for the ExploratoryDataAnalysis class docstring"""
        self.assertIsNot(ExploratoryDataAnalysis.__doc__, None,
                         "ExploratoryDataAnalysis class needs a docstring")
        self.assertTrue(len(ExploratoryDataAnalysis.__doc__) >= 1,
                        "ExploratoryDataAnalysis class needs a docstring")

    def test_func_docstrings(self):
        """
        Test for the presence of docstrings in ExploratoryDataAnalysis methods.
        """
        for func in self.base_funcs:
            with self.subTest(function=func):
                self.assertIsNot(
                    func[1].__doc__,
                    None,
                    f"{func[0]} method needs a docstring"
                )
                self.assertTrue(
                    len(func[1].__doc__) > 1,
                    f"{func[0]} method needs a docstring"
                )


class TestExploratoryDataAnalysis(unittest.TestCase):

    @patch('src.exploratory_data_analysis.create_engine')
    def setUp(self, mock_create_engine):
        # Mock the SQLAlchemy engine
        self.mock_engine = MagicMock()
        mock_create_engine.return_value = self.mock_engine

        # Initialize the ExploratoryDataAnalysis object
        self.db_url = 'sqlite:///:memory:'
        self.eda = ExploratoryDataAnalysis(self.db_url)

    def test_init(self):
        # Test if the engine is created properly
        self.assertEqual(self.eda.engine, self.mock_engine)

    @patch('pandas.read_sql')
    def test_load_data(self, mock_read_sql):
        # Mock the data returned by read_sql
        mock_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        mock_read_sql.return_value = mock_data

        # Call the load_data method
        table_name = 'test_table'
        result = self.eda.load_data(table_name)

        # Assert read_sql was called with the correct parameters
        mock_read_sql.assert_called_once_with(
            f"SELECT * FROM {table_name}", self.mock_engine)

        # Assert the result is as expected
        pd.testing.assert_frame_equal(result, mock_data)

    @patch('pandas.DataFrame.describe')
    @patch('pandas.DataFrame.dtypes')
    def test_data_summarization(self, mock_dtypes, mock_describe):
        # Mock the DataFrame describe method
        mock_describe.return_value = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        # Mock the DataFrame dtypes attribute
        mock_dtypes.return_value = pd.Series(
            {'col1': 'int64', 'col2': 'object'})

        # Create a sample DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        # Call the data_summarization method
        descriptive_stats, data_structure = self.eda.data_summarization(df)

        # Assert the describe method was called
        mock_describe.assert_called_once()

        # Assert the dtypes attribute was accessed
        self.assertFalse(mock_dtypes.called)

        # Assert the results are as expected
        pd.testing.assert_frame_equal(
            descriptive_stats, mock_describe.return_value)
        """
        pd.testing.assert_series_equal(
            data_structure, mock_dtypes.return_value)
        """
    @patch(
        'src.exploratory_data_analysis.ExploratoryDataAnalysis.' +
        'correlation_and_plot')
    def test_univariate_analysis(self, mock_plotting_function):
        # Mock the plotting function
        mock_plotting_function.return_value = 'mock_plot'

        # Create a sample DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        # Call the univariate_analysis method
        univariate_plots, univariate_data = self.eda.univariate_analysis(df)

        # Assert the plotting function was called
        # self.assertTrue(mock_plotting_function.called)

        # Assert the results are as expected
        # self.assertEqual(univariate_plots, 'mock_plot')
        # self.assertEqual(univariate_data, df.describe())

    @patch('src.exploratory_data_analysis.ExploratoryDataAnalysis.' +
           'create_line_plot')
    @patch('src.exploratory_data_analysis.ExploratoryDataAnalysis.' +
           'create_line_plot')
    def test_bivariate_multivariate_analysis(
            self, mock_scatter_plot_function, mock_correlation_plot_function):
        # Mock the plotting functions
        mock_scatter_plot_function.return_value = 'mock_scatter_plot'
        mock_correlation_plot_function.return_value = 'mock_correlation_plot'

        # Create a sample DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })

        # Call the bivariate_multivariate_analysis method
        # corr_matrix, plots = self.eda.bivariate_multivariate_analysis(df)

        # Assert the plotting functions were called
        # self.assertTrue(mock_scatter_plot_function.called)
        # self.assertTrue(mock_correlation_plot_function.called)

        # Assert the results are as expected
        # self.assertEqual(correlation_matrix, df.corr())
        # self.assertEqual(plots, ('mock__plot', 'mock_scatter_plot'))


if __name__ == '__main__':
    unittest.main()
