# BIOMS: Binary Integrals of Motion

BIOMS is a small python package for constructing quantum operators that are approximate binary integrals of motion. It is built on top of the [QOSY python package](https://github.com/ClarkResearchGroup/qosy).

## Features

BIOMS implements a heuristic algorithm that takes as input an operator H (such as a Hamiltonian) and produces as output an operator O that minimizes the sum of the binarity |O^2 - I|^2 and the commutator norm |[H, O]|^2.

## Getting Started

### Prerequisites

BIOMS requires the following software installed on your platform:
- [Python 3](https://www.python.org/)
- [Numpy](https://www.numpy.org/)
- [Scipy](https://www.scipy.org/)
- [QOSY](https://github.com/ClarkResearchGroup/qosy)
- If you want to run the tests: [pytest](https://pytest.org)

### Installing

To copy the development version of the code to your machine, type
```
git clone https://github.com/ClarkResearchGroup/bioms.git
```
To install, type
```
cd bioms
python setup.py install --user
```
or add the bioms folder to your PYTHONPATH environment variable.

## Testing

To test BIOMS after installation (recommended), type
```
cd bioms/tests
pytest
```

## Authors

- Eli Chertkov

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) for details.

## References

BIOMS is introduced in the following study of many-body localization in greater than one dimensions:

...

Qosy is based on work presented in

E. Chertkov, B. Villalonga, and B. K. Clark, “Engineering Topological Models with a General-Purpose Symmetry-to-Hamiltonian Approach,” [arXiv:1910.10165](http://arxiv.org/abs/1910.10165) (2019).
