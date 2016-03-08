SORTED_FLAG ?= -DUNSORTED
UNION_FLAG ?= -DUNION_PARALLEL_SPA
INTERSECT_FLAG ?= -DINTERSECT_PARALLEL_SPA
SPGEMM_FLAG ?= -DSPGEMM_PARALLEL_SPA
FSPGEMM_FLAG ?= -DFSPGEMM_PARALLEL_SPA

#SORTED_FLAG ?= -DSORTED
#UNION_FLAG ?= -DUNION_PARALLEL_MERGE
#INTERSECT_FLAG ?= -DINTERSECT_PARALLEL_MERGE
#SPGEMM_FLAG ?= -DSPGEMM_PARALLEL_MERGE
#FSPGEMM_FLAG ?= -DFSPGEMM_PARALLEL_SPA

GPSOURCES = $(GRAPHPAD_PATH)/src/graphpad.h $(GRAPHPAD_PATH)/src/SpMat.h $(GRAPHPAD_PATH)/src/singlenode_spgemm.h $(GRAPHPAD_PATH)/src/singlenode_intersectreduce.h $(GRAPHPAD_PATH)/src/singlenode_unionreduce.h $(GRAPHPAD_PATH)/src/singlenode_reduce.h $(GRAPHPAD_PATH)/src/multinode_reduce.h $(GRAPHPAD_PATH)/src/multinode_intersectreduce.h $(GRAPHPAD_PATH)/src/multinode_spgemm.h  $(GRAPHPAD_PATH)/src/multinode_unionreduce.h $(GRAPHPAD_PATH)/src/multinode_fspgemm.h $(GRAPHPAD_PATH)/src/singlenode_fspgemm.h $(GRAPHPAD_PATH)/src/CSRTile.h $(GRAPHPAD_PATH)/src/DCSCTile.h $(GRAPHPAD_PATH)/src/SpVec.h $(GRAPHPAD_PATH)/src/DenseSegment.h $(GRAPHPAD_PATH)/src/multinode_spmspv.h $(GRAPHPAD_PATH)/src/DenseTile.h $(GRAPHPAD_PATH)/src/TileOps.h $(GRAPHPAD_PATH)/src/edgelist.h $(GRAPHPAD_PATH)/src/singlenode_spmspv.h $(GRAPHPAD_PATH)/src/singlenode_apply.h $(GRAPHPAD_PATH)/src/multinode_apply.h $(GRAPHPAD_PATH)/src/multinode_clear.h $(GRAPHPAD_PATH)/src/singlenode_clear.h $(GRAPHPAD_PATH)/src/multinode_spmspv3.h $(GRAPHPAD_PATH)/src/singlenode_spmspv3.h $(GRAPHPAD_PATH)/src/bitvector.h $(GRAPHPAD_PATH)/src/multinode_sparsify.h $(GRAPHPAD_PATH)/src/singlenode_sparsify.h $(GRAPHPAD_PATH)/src/CSSegment.h $(GRAPHPAD_PATH)/src/multinode_fspmspv.h $(GRAPHPAD_PATH)/src/singlenode_fspmspv.h $(GRAPHPAD_PATH)/src/layouts.h $(GRAPHPAD_PATH)/src/binary_search.h

GPFLAGS = $(UNION_FLAG) $(INTERSECT_FLAG) $(SPGEMM_FLAG) $(EDGEWEIGHT_FLAG) $(SORTED_FLAG) $(FSPGEMM_FLAG)

