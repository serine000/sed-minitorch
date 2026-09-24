# Sed minitorch

This is my own minitorch implementation.
It's a smaller more basic version of the Torch library


## Description
This is a minitiature version implementation of the Torch library.
It starts from the basic python library operations and higher order functions.

## Running tests
For running test suites under a specific marker, run:
`pytest FILE_NAME -m MARKER`

## Running Streamlit
For running streamlit:
- Make sure you're on the repository root dir: `sed-minitorch/` and not inside `project/` when trying to run streamlit
- Run `python -m streamlit run project/app.py -- 0`


## Changes done
These are some of the functional changes I did to the original codes because of outdated errors in the original repo:
(You can find the corrected plain version of this repo to follow the guides with on <>.)
- Replace any instance of `st.cache` with `st.cache_data`
- Modern Streamlit gives elements IDs based on their type/parameters and rejects duplicate IDs unless you provide unique key= values.


## Concepts included
- Chain rule
- Scalar abstraction
- Backpropagation

