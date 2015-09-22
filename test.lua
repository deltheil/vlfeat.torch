require 'totem'
require 'vlfeat'

local tests  = {}
local tester = totem.Tester()

local N      = 100
local D      = 64
local K      = 4

function tests.vl_kmeans_smoke_tests_1()
  local data = torch.rand(N, D):double()
  local kmeans = vlfeat.KMeans()

  tester:asserteq(kmeans:algorithm(), 'LLOYD')
  tester:asserteq(kmeans:initialization(), 'RAND')
  tester:asserteq(kmeans:numRepetitions(3), 3)

  kmeans:cluster(data, K)

  local centers = kmeans:centers()
  tester:asserteq(centers:dim(), 2)
  tester:asserteq(centers:size(1), K)
  tester:asserteq(centers:size(2), D)

  local indices = torch.LongTensor{1, 2, 4}
  local points = centers:index(1, torch.LongTensor{1, 2, 4})
  local assignments, distances = kmeans:quantize(points)
  tester:assertTensorEq(assignments, indices:int())
  tester:assertTensorEq(distances, torch.DoubleTensor(3):fill(0), 1e-4)
end

function tests.vl_kmeans_smoke_tests_2()
  local data = torch.rand(N, D):float()
  local kmeans = vlfeat.KMeans('L1'):float()

  tester:asserteq(kmeans:algorithm('ELKAN'), 'ELKAN')
  tester:asserteq(kmeans:initialization('PLUSPLUS'), 'PLUSPLUS')
  tester:asserteq(kmeans:maxIter(150), 150)

  kmeans:initCentersWithRandData(data, K)
  kmeans:refineCenters(data)

  local centers = kmeans:centers()
  tester:asserteq(torch.typename(centers), 'torch.FloatTensor')
end

function tests.vl_kmeans_smoke_tests_3()
  local data = torch.rand(N, D)
  local centers = torch.rand(K, D)

  local kmeans = vlfeat.KMeans()

  kmeans:setCenters(centers)

  tester:assertTensorEq(kmeans:centers(), centers)
end

tester:add(tests)
return tester:run(tests)
