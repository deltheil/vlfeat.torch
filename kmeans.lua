local KMeans = torch.class('vlfeat.KMeans')
local ffi    = require 'ffi'
local C      = vlfeat.C
local NULL   = vlfeat.NULL

local VL_KMEANS_ALGO = {
  LLOYD                  = C.VlKMeansLloyd,
  ELKAN                  = C.VlKMeansElkan,
  ANN                    = C.VlKMeansANN
}

local VL_KMEANS_INIT = {
  RAND                   = C.VlKMeansRandomSelection,
  PLUSPLUS               = C.VlKMeansPlusPlus
}

local assert_valid = function(self, tensor)
  local th = torch.type(tensor)
  local vl = self.handle.dataType

  assert(
    vlfeat.TH_VL_TYPE[th] == vl,
    'expected ' .. vlfeat.VL_TH_TYPE[vl] .. ', got ' .. th
  )

  assert(
    tensor:dim() == 2,
    'expected 2D tensor, got ' .. tensor:dim() .. 'D tensor'
  )

  assert(
    tensor:isContiguous(),
    'expected a contiguous tensor'
  )
end

function KMeans:__init(dist)
  if dist then
    assert(
      type(dist) == 'string',
      'expected `string`, got `' .. type(dist) .. '`'
    )
    dist = assert(
      vlfeat.VECT_COMPARISON_TYPE[string.upper(dist)],
      'unsupported distance: `' .. dist .. '`'
    )
  else
    dist = C.VlDistanceL2
  end
  self.handle = ffi.gc(
    C.vl_kmeans_new(C.VL_TYPE_DOUBLE, dist),
    C.vl_kmeans_delete
  )
end

function KMeans:float()
  self.handle.dataType = C.VL_TYPE_FLOAT
  return self
end

function KMeans:double()
  self.handle.dataType = C.VL_TYPE_DOUBLE
  return self
end

function KMeans:algorithm(value)
  if value then
    assert(
      type(value) == 'string',
      'expected `string`, got `' .. type(value) .. '`'
    )
    value = string.upper(value)
    self.handle.algorithm = assert(
      VL_KMEANS_ALGO[value],
      'unsupported algorithm: `' .. value .. '`'
    )
  else
    for k,v in pairs(VL_KMEANS_ALGO) do
      if v == self.handle.algorithm then
        value = k; break
      end
    end
    assert(value)
  end
  return value
end

function KMeans:initialization(value)
  if value then
    assert(
      type(value) == 'string',
      'expected `string`, got `' .. type(value) .. '`'
    )
    value = string.upper(value)
    self.handle.initialization = assert(
      VL_KMEANS_INIT[value],
      'unsupported initialization: `' .. value .. '`'
    )
  else
    for k,v in pairs(VL_KMEANS_INIT) do
      if v == self.handle.initialization then
        value = k; break
      end
    end
    assert(value)
  end
  return value
end

function KMeans:maxIter(value)
  if value then
    assert(
      type(value) == 'number',
      'expected `number`, got `' .. type(value) .. '`'
    )
    self.handle.maxNumIterations = ffi.cast('vl_size', value)
  else
    value = tonumber(self.handle.maxNumIterations)
  end
  return value
end

function KMeans:numRepetitions(value)
  if value then
    assert(
      type(value) == 'number',
      'expected `number`, got `' .. type(value) .. '`'
    )
    self.handle.numRepetitions = ffi.cast('vl_size', value)
  else
    value = tonumber(self.handle.numRepetitions)
  end
  return value
end

function KMeans:verbosity(value)
  if value then
    assert(
      type(value) == 'number',
      'expected `number`, got `' .. type(value) .. '`'
    )
    self.handle.verbosity = value
  else
    value = tonumber(self.handle.verbosity)
  end
  return value
end

function KMeans:cluster(tensor, numCenters)
  assert_valid(self, tensor)
  return C.vl_kmeans_cluster(
    self.handle,
    tensor:data(),
    tensor:size(2),
    tensor:size(1),
    numCenters
  )
end

function KMeans:initCentersWithRandData(tensor, numCenters)
  assert_valid(self, tensor)
  local init_centers
  if self:initialization() == 'PLUSPLUS' then
    init_centers = C.vl_kmeans_init_centers_plus_plus
  else
    init_centers = C.vl_kmeans_init_centers_with_rand_data
  end
  return init_centers(
    self.handle,
    tensor:data(),
    tensor:size(2),
    tensor:size(1),
    numCenters
  )
end

function KMeans:setCenters(tensor)
  assert_valid(self, tensor)
  return C.vl_kmeans_set_centers(
    self.handle,
    tensor:data(),
    tensor:size(2),
    tensor:size(1)
  )
end

function KMeans:refineCenters(tensor)
  assert_valid(self, tensor)
  return C.vl_kmeans_refine_centers(
    self.handle,
    tensor:data(),
    tensor:size(1)
  )
end

function KMeans:centers()
  assert(
    self.handle.centers ~= NULL,
    'NULL centers'
  )
  local centers = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tonumber(self.handle.numCenters),
    tonumber(self.handle.dimension)
  )
  assert(centers:isContiguous())
  ffi.copy(
    centers:data(),
    self.handle.centers,
    centers:nElement() * centers:storage():elementSize()
  )
  return centers
end

function KMeans:quantize(tensor)
  assert_valid(self, tensor)
  local assignments = torch.IntTensor(tensor:size(1))
  local distances   = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tensor:size(1)
  )
  C.vl_kmeans_quantize(
    self.handle,
    assignments:data(),
    distances:data(),
    tensor:data(),
    tensor:size(1)
  )
  return assignments:add(1), distances
end
