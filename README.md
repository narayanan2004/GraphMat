GraphMat
========

GraphMat graph analytics framework

Requirements:
Intel compiler (icpc).

To compile:
do make in the GraphMat/ directory

To run:
set the following environment variables:
export OMP_NUM_THREADS=<number of cores in system>
export KMP_AFFINITY=scatter

use numactl for NUMA (multi-socket) systems e.g.
numactl -i all bin/PageRank <graph file> 
numactl -i all bin/BFS <graph file> <start vertex>

You can convert an edge list file to GraphMat compatible format using the graph_converter utility. GraphMat works on a binary graph format.


If you use GraphMat in your work, please cite the following paper:
Narayanan Sundaram, Nadathur Satish, Md Mostofa Ali Patwary, Subramanya R Dulloor, Michael J. Anderson, Satya Gautam Vadlamudi, Dipankar Das, Pradeep Dubey:
GraphMat: High performance graph analytics made productive. Proceedings of VLDB 2015, volume 8, pages 1214 - 1225.

PDF available at
www.vldb.org/pvldb/vol8/p1214-sundaram.pdf


More documentation coming soon.
For questions, please email narayanan.sundaram@intel.com

