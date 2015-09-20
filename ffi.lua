local ffi = require 'ffi'

-- host specific typedef-s
if ffi.abi('64bit') then
  ffi.cdef[[
typedef long long unsigned  vl_uint64;
typedef int       unsigned  vl_uint32;
typedef vl_uint64           vl_size;
]]
else
  ffi.cdef[[
typedef int       unsigned  vl_uint32;
typedef vl_uint32           vl_size;
]]
end

ffi.cdef[[
typedef vl_uint32 vl_type;

enum {
  VL_TYPE_FLOAT = 1,
  VL_TYPE_DOUBLE
};

typedef float (*VlFloatVectorComparisonFunction)(
  vl_size dimension, float const * X, float const * Y
);
typedef double (*VlDoubleVectorComparisonFunction)(
  vl_size dimension, double const * X, double const * Y
);

typedef enum _VlKMeansAlgorithm {
  VlKMeansLloyd,
  VlKMeansElkan,
  VlKMeansANN
} VlKMeansAlgorithm;

typedef enum _VlKMeansInitialization {
  VlKMeansRandomSelection,
  VlKMeansPlusPlus
} VlKMeansInitialization;

enum _VlVectorComparisonType {
  VlDistanceL1,
  VlDistanceL2,
  VlDistanceChi2,
  VlDistanceHellinger,
  VlDistanceJS,
  VlDistanceMahalanobis,
  VlKernelL1,
  VlKernelL2,
  VlKernelChi2,
  VlKernelHellinger,
  VlKernelJS
};

typedef enum _VlVectorComparisonType VlVectorComparisonType;

typedef struct _VlKMeans
{

  vl_type dataType;
  vl_size dimension;
  vl_size numCenters;
  vl_size numTrees;
  vl_size maxNumComparisons;
  VlKMeansInitialization initialization;
  VlKMeansAlgorithm algorithm;
  VlVectorComparisonType distance;
  vl_size maxNumIterations;
  double minEnergyVariation;
  vl_size numRepetitions;
  int verbosity;
  void * centers;
  void * centerDistances;
  double energy;
  VlFloatVectorComparisonFunction floatVectorComparisonFn;
  VlDoubleVectorComparisonFunction doubleVectorComparisonFn;
} VlKMeans;

VlKMeans * vl_kmeans_new(
  vl_type dataType,
  VlVectorComparisonType distance
);

void vl_kmeans_delete(
  VlKMeans * self
);

double vl_kmeans_cluster(
  VlKMeans * self,
  void const * data,
  vl_size dimension,
  vl_size numData,
  vl_size numCenters
);

void vl_kmeans_set_centers(
  VlKMeans * self,
  void const * centers,
  vl_size dimension,
  vl_size numCenters
);

void vl_kmeans_init_centers_with_rand_data(
  VlKMeans * self,
  void const * data,
  vl_size dimensions,
  vl_size numData,
  vl_size numCenters
);

void vl_kmeans_init_centers_plus_plus(
  VlKMeans * self,
  void const * data,
  vl_size dimensions,
  vl_size numData,
  vl_size numCenters
);

double vl_kmeans_refine_centers(
  VlKMeans * self,
  void const * data,
  vl_size numData
);

void vl_kmeans_quantize(
  VlKMeans * self,
  vl_uint32 * assignments,
  void * distances,
  void const * data,
  vl_size numData
);
]]

-- RTLD_GLOBAL mode (cf. man 3 dlopen)
local global = true

local ok, C = pcall(ffi.load, 'vl', global)
if not ok then
   error('cannot load VLFeat library: ' .. C)
end
vlfeat.C = C

vlfeat.TH_VL_TYPE = {
  ['torch.FloatTensor']  = C.VL_TYPE_FLOAT,
  ['torch.DoubleTensor'] = C.VL_TYPE_DOUBLE
}

vlfeat.VL_TH_TYPE = {
  [C.VL_TYPE_FLOAT]      = 'torch.FloatTensor',
  [C.VL_TYPE_DOUBLE]     = 'torch.DoubleTensor'
}

vlfeat.VECT_COMPARISON_TYPE = {
  L1                     = C.VlDistanceL1,
  L2                     = C.VlDistanceL2,
  CHI2                   = C.VlDistanceChi2,
  HELLINGER              = C.VlDistanceHellinger,
  JS                     = C.VlDistanceJS,
  MAHALANOBIS            = C.VlDistanceMahalanobis,
  KERNELL1               = C.VlKernelL1,
  KERNELL1               = C.VlKernelL2,
  KERNELCHI2             = C.VlKernelChi2,
  KERNELHELLINGER        = C.VlKernelHellinger,
  KERNELJS               = C.VlKernelJS
}
