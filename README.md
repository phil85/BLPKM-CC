# BLPKM-CC

Constrained clustering algorithm that considers must-link and cannot-link constraints

## Dependencies

BLPKM-CC depends on:
* [Gurobi](https://github.com/spyder-ide/spyder)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Clone this repository (git clone https://github.com/phil85/BLPKM-CC.git)

## Usage

The main.py file contains code that applies the BLPKM-CC algorithm on an illustrative example.

```python
labels = blpkm_cc(data=X, n_clusters=2)
```

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P.** (2020): Clustering with Must-Link and Cannot-Link Constraints: Comparing BLPKM to DILS. Proceedings of the 2020 IEEE International Conference on Industrial Engineering and Engineering Management. Singapore, to appear

Bibtex:
```
@inproceedings{baumann2020clustering,
	author={Philipp Baumann},
	booktitle={2020 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM)},
	title={Clustering with Must-Link and Cannot-Link Constraints: Comparing BLPKM to DILS},
	year={2020},
	volume={},
	number={},
	pages={to appear},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

