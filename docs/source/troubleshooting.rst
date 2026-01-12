Troubleshooting Guide
=====================

Common issues and solutions when using PlotSmith.

Installation Issues
-------------------

**Issue**: Import errors or missing dependencies

**Solution**: Ensure all dependencies are installed:

.. code-block:: bash

   pip install plotsmith[all]

**Issue**: Python version compatibility

**Solution**: PlotSmith requires Python 3.12+. Check your Python version:

.. code-block:: bash

   python --version

Plotting Issues
---------------

**Issue**: Plots not displaying or saving

**Solution**: Ensure matplotlib backend is set correctly:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # For non-interactive backends
   # or
   matplotlib.use('TkAgg')  # For GUI backends

**Issue**: Empty plots or no data visible

**Solution**: Check that your data is not empty and has valid values:

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Check data
   print(data.head())
   print(data.info())
   print(data.isna().sum())

**Issue**: "Column not found" errors

**Solution**: Verify column names match exactly (case-sensitive):

.. code-block:: python

   # Check available columns
   print(df.columns.tolist())
   
   # Use exact column names
   plot_scatter(df, x='exact_column_name', y='another_column')

Data Validation Issues
-----------------------

**Issue**: "Mismatched lengths" errors

**Solution**: Ensure all arrays/Series have the same length:

.. code-block:: python

   # Check lengths
   print(len(x), len(y))
   
   # Align data if needed
   common_index = x.index.intersection(y.index)
   x_aligned = x.loc[common_index]
   y_aligned = y.loc[common_index]

**Issue**: NaN values causing problems

**Solution**: Handle NaN values before plotting:

.. code-block:: python

   # Remove NaN values
   data_clean = data.dropna()
   
   # Or fill NaN values
   data_filled = data.fillna(0)

**Issue**: Empty data errors

**Solution**: Check for empty data before plotting:

.. code-block:: python

   if len(data) == 0:
       print("Warning: Data is empty")
   else:
       fig, ax = plot_timeseries(data)

Type Errors
-----------

**Issue**: "TypeError: data must be pandas Series or DataFrame"

**Solution**: Convert your data to the correct type:

.. code-block:: python

   # Convert list to Series
   data = pd.Series([1, 2, 3])
   
   # Convert numpy array to Series
   data = pd.Series(array_data, index=pd.date_range('2020-01-01', periods=len(array_data)))

**Issue**: Type checking errors with mypy

**Solution**: PlotSmith uses type hints. Ensure you're using compatible types:

.. code-block:: python

   from plotsmith import plot_timeseries
   import pandas as pd
   
   # Correct: pandas Series
   data: pd.Series = pd.Series([1, 2, 3])
   fig, ax = plot_timeseries(data)

Performance Issues
------------------

**Issue**: Slow plotting with large datasets

**Solution**: 
- Use data sampling for very large datasets
- Consider using vectorized operations
- Close figures explicitly to free memory

.. code-block:: python

   # Sample large data
   if len(data) > 10000:
       data_sample = data.sample(1000)
       fig, ax = plot_timeseries(data_sample)
   else:
       fig, ax = plot_timeseries(data)
   
   # Always close figures
   plt.close(fig)

Styling Issues
--------------

**Issue**: Styling not applied correctly

**Solution**: Ensure you're calling styling functions after creating the plot:

.. code-block:: python

   from plotsmith import plot_timeseries, minimal_axes
   
   fig, ax = plot_timeseries(data)
   minimal_axes(ax)  # Apply styling after plot creation

**Issue**: Fonts not displaying correctly

**Solution**: Ensure matplotlib can find the fonts:

.. code-block:: python

   import matplotlib
   print(matplotlib.get_configdir())
   # Check font cache: matplotlib.font_manager._rebuild()

CI/CD Issues
------------

**Issue**: Tests failing in CI

**Solution**: 
- Use non-interactive backend: `matplotlib.use('Agg')`
- Ensure all dependencies are installed
- Check Python version compatibility

**Issue**: Coverage below threshold

**Solution**: Add tests for uncovered code paths:

.. code-block:: bash

   pytest --cov=plotsmith --cov-report=html
   # Review htmlcov/index.html for coverage gaps

Getting Help
------------

If you encounter issues not covered here:

1. **Check Documentation**: Review the full documentation
2. **Search Issues**: Check GitHub issues for similar problems
3. **Create Issue**: Open a new issue with:
   - Minimal reproducible example
   - Error messages
   - Python and PlotSmith versions
   - Operating system

Example Issue Template:

.. code-block:: python

   import pandas as pd
   from plotsmith import plot_timeseries
   
   # Your code here
   data = pd.Series([1, 2, 3])
   fig, ax = plot_timeseries(data)  # Error occurs here

