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
- `index_to_position`: This function is about getting a multidimensional index (i.e. (1, 2)) and converting it to a single numerical index that we can use to access that same piece of data but in its position inside the storage of the tensor. So Tensor(1, 2) could map to storage[5].
- `to_index`: This function does the actual opposite of `index_to_position` and it actually takes an ordinal number and has to transform it back to a multidimensional index.

[Important note: They're not _exactly_ opposites since `to_index` uses shape, while `index_to_position` uses strides.]

- `permute`: THe purpose of this function is to rearrange the dimensions of a tensor without rearranging the actual data in storage. This operation is equivalent to _transposing_ a matrix. When we permute, we generate a new shape and stride for our data. We also change the stride because each stride belongs to a specific dimension. When that dimension moves, its stride must move with it.

### Broadcasting
Broadcasting in general is about performing an operation between two tensors of different shapes. It allows us to perform these operations between differently shaped tensors without explicitly copying or expanding their data.

- `shape_broadcast`: This functions helps determine the resulting shape of two different tensors once we union them (And when comparing two shapes, we align them from the rightmost dimension.)
  - For each pair of dimensions, broadcasting is allowed if:
  1. The dimensions are equal.
  2. At least one dimension is 1.
  For the shorter dimension, since we're strating from the rightmost dimension, any missing ones will be padded on the left with 1s. (e.g. (2, 3) and (3,) become (2, 3) and (1, 3)).

- `broadcast_index`: This function determines which element of an original tensor corresponds to a particular index in the broadcasted result.
  - For example: Say we want to do -> C[1, 2] = A[1, 2] + B[?] but we can't use [1, 2] with B since it's 1-dimensional. So we need to find what [1, 2] corresponds in B.

#### Putting it all together
```Python
C.shape = (2, 3)

for ordinal in range(C.size): # Size = 6 so we go from 0 to 5
    # Convert ordinal to output tensor index
    to_index(ordinal, C.shape, c_index)

    # Find corresponding indices in A and B
    broadcast_index(c_index, C.shape, A.shape, a_index)
    broadcast_index(c_index, C.shape, B.shape, b_index)

    # Find their physical storage positions
    a_pos = index_to_position(a_index, A.strides)
    b_pos = index_to_position(b_index, B.strides)

    # Compute output value
    C.storage[ordinal] = A.storage[a_pos] + B.storage[b_pos]
```

### Tensor Operations
