## vlfeat.torch

VLFeat (partial) FFI wrapper for Torch7.

### Install

```bash
luarocks make
```

### Example

K-means clustering:

```Lua
local vlfeat = require 'vlfeat'

local N   = 100
local D   = 64
local K   = 4

local kmeans = vlfeat.KMeans()

-- generate some random data (N x D-dimensional points)
local data = torch.rand(N, D)

-- run the algorithm with K centers (a.k.a. clusters)
kmeans:cluster(data, K)

-- get computed centers
local centers = kmeans:centers()
assert(
  (centers:size(1) == K) and
  (centers:size(2) == D)
)

-- pick some centers (for test purpose) and retrieve the corresponding indices
local points = centers:index(1, torch.LongTensor{1, 2, 4})
local assignments = kmeans:quantize(points)
assert(
  (assignments[1] == 1) and
  (assignments[2] == 2) and
  (assignments[3] == 4)
)
```
