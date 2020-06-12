# BLPKM-CC

Constrained clustering algorithm that considers must-link and cannot-link constraints. 

## Dependencies

BLPKM-CC depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). A version of this algorithm that uses the non-commercial [SCIP](https://www.scipopt.org/) solver is available upon request. Please contact me by email (philipp.baumann@pqm.unibe.ch) if you are interested.

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Clone this repository (git clone https://github.com/phil85/BLPKM-CC.git)

## Usage

The main.py file contains code that applies the BLPKM-CC algorithm on an illustrative example.

```python
labels = blpkm_cc(X, n_clusters=2, ml=ml, cl=cl)
```

## Documentation

The documentation of the module blpkm_cc can be found [here](https://phil85.github.io/BLPKM-CC/documentation.html).

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P.** (2020): Clustering with Must-Link and Cannot-Link Constraints: Comparing BLPKM to DILS. Proceedings of the 2020 IEEE International Conference on Industrial Engineering and Engineering Management. Singapore, under review

Bibtex:
```
@inproceedings{baumann2020clustering,
	author={Philipp Baumann},
	booktitle={2020 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM)},
	title={Clustering with Must-Link and Cannot-Link Constraints: Comparing BLPKM to DILS},
	year={2020},
	volume={},
	number={},
	pages={under review},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


