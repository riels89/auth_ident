#!/bin/bash

mkdir -p data/loaded
python src/data_processing_expt/save_hdf.py --val-test-split .1 --out data/loaded/python &
python src/data_processing_expt/save_hdf.py --extensions cxx cc cpp c++ C --val-test-split .1 --out data/loaded/cpp &
python src/data_processing_expt/save_hdf.py --extensions c cxx cc cpp c++ C --val-test-split .1 --out data/loaded/c_cpp &
python src/data_processing_expt/save_hdf.py --extensions java --val-test-split .1 --out data/loaded/java &
python src/data_processing_expt/save_hdf.py --extensions c cxx cc cpp c++ C h hh H hxx hpp h++ --val-test-split .1 --out data/loaded/c_cpp_h
