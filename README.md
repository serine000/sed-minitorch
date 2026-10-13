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
- Topological sort

## Steps summarized
- Defining basic operations
- Defining higher order utility functions (from scratch)
- Defining Modules and Parameters
- Implementing differentiation
- Defining Scalar & ScalarHistory + operation Context
- Defining the forward and backward of every operation
- Implementing topological sort ordering
- Implementing the Chain Rule mechanism
- Imeplementing Backpropagation
- Tensor operations implementation

## Tensor operations

### Indexing
For indexing we want to be able to move around the tensor and fetch our data from the underlying storage.
- `index_to_position` is about getting a multidimensional index (i.e. (1, 2)) and converting it to a single numerical index that we can use to access that same piece of data but in its position inside the storage of the tensor. So Tensor(1, 2) could map to storage[5].
- `to_index` does the actual opposite of `index_to_position` and it actually takes an ordinal number and has to transform it back to a multidimensional index.
[Important note: They're not _exactly_ opposites since `to_index` uses shape, while `index_to_position` uses strides.]
