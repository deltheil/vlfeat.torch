require 'torch' -- for `include` and class system
vlfeat = {}     -- top-level module for Torch class system (it MUST be global!)

include 'ffi.lua'
include 'kmeans.lua'

return vlfeat
