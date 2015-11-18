# GraphMat graph analytics framework

Requirements:

- Intel compiler (icpc)

To compile:

    make

To run:

Set the following environment variables:

    export OMP_NUM_THREADS=[ number of cores in system ]
    export KMP_AFFINITY=scatter

Use `numactl` for NUMA (multi-socket) systems e.g.

    numactl -i all bin/PageRank < graph file >
    numactl -i all bin/BFS < graph file > < start vertex >

You can convert an edge list file to GraphMat compatible format using
the `graph_converter` utility. GraphMat works on a binary graph
format.

To convert from a text file with 3 white space separated columns
(`src, dst, edge_value`) to GraphMat format, do

    bin/graph_converter --selfloops 1 --duplicatededges 1 --inputformat 1 --outputformat 0 --inputheader 0 --outputheader 1 --nvertices < nvertices > < input text file > < output graphmat file >

You can remove selfloops and duplicatededges (when multiple edges with
same src and dst are found, only one is retained) by changing their
values in the command line from 1 to 0.

If you use GraphMat in your work, please cite the following paper:

Narayanan Sundaram, Nadathur Satish, Md Mostofa Ali Patwary, Subramanya R Dulloor, Michael J. Anderson, Satya Gautam Vadlamudi, Dipankar Das, Pradeep Dubey:
GraphMat: High performance graph analytics made productive. Proceedings of VLDB 2015, volume 8, pages 1214 - 1225.

Paper PDF: www.vldb.org/pvldb/vol8/p1214-sundaram.pdf

More documentation coming soon. For questions, please email narayanan.sundaram@intel.com
