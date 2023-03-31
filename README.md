# Modification of "Impacts of Uncertainty & Differential Privacy on Title I" To perform Shrinkage Estimates 

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Author: Kentaro Hoffman, Ruobin Gong, Yifan Cui, and Jan Hannig

A Fork of ryansteed/dp-policy written by: [Ryan Steed](rbsteed.com), with help from Terrance Liu

API documentation for ryansteed/dp-policy [https://github.com/ryansteed/dp-policy] : [rbsteed.com/dp-policy](https://rbsteed.com/dp-policy/)

For more information about ryansteed/dp-policy , check out the [paper](https://rbsteed.com/referral/dp-policy) and [SI](https://www-science-org.cmu.idm.oclc.org/doi/suppl/10.1126/science.abq4481/suppl_file/science.abq4481_sm.pdf).

## Installation

To install, follow the same installation instructions as found on ryansteed/dp-policy [https://github.com/ryansteed/dp-policy]


## To Run

To perform the allocation estimates with shrinkage, open "notebooks/Shrunk_Titlei.ipynb" and run the file. This will populate the following folders:

- notebooks/Gaussian (Gaussian Error Scheme)  
- notebooks/H_Prop (Morris-Lysy)
- notebooks/HB (Hudson-Berger Shrinkage Scheme)

Each folder will contain files:

- (Shrinkage Scheme)_allocations.csv 
- (Shrinkage Scheme)_allocations_dp.csv 

(Shrinkage Scheme)_allocations.csv represents a matrix where the entires are the estimated Title i allocations that were performed if one shrunk the population counts and allocated based on those shrunk counts. (Shrinkage Sccheme)_allocations_dp.csv is similar however it estimates the Title i allocations if one shrunk the population counts, added gaussian differnetial privacy noise and allocated based on the result. 

An RMarkdown file analyzing these files can be found in notebooks/JS_Compairson.Rmd.

