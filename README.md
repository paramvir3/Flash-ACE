# Flash-ACE
This repository is an attempt to make use attention mechanism on Atomic Cluster Expansion (Drautz, Phys. Rev. B 99, 2019) for making precise and scalable machine learning interatomic potentials

Please DO NOT USE as this is purely for research purposes

## Running the ACE descriptor test

The regression test added for the ACE descriptor verifies that the output depends on
per-atom attributes (chemical identity-like information) as in the MACE implementation.

1. Install dependencies (PyTorch, e3nn, torch-scatter and other packages declared in
   `setup.py`). A minimal CPU-only setup can be installed with:

   ```bash
   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   pip install -e .
   ```

2. Run the specific test:

   ```bash
   pytest tests/test_ace_descriptor.py -q
   ```

When the test passes it exits silently with status code 0. The assertion inside the
test checks that modifying a node attribute changes the descriptor values while keeping
the tensor shape unchanged, confirming the descriptor construction matches the MACE
behavior.


