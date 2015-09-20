package = "vlfeat"
version = "scm-1"

source = {
   url = "https://github.com/deltheil/vlfeat.torch.git",
}

description = {
   summary = "VLFeat (partial) FFI wrapper for Torch7",
   detailed = [[
LuaJIT FFI interface to VLFeat suitable for Torch7. It is not intended to be
exhaustive in terms of API coverage.
   ]],
   homepage = "https://github.com/deltheil/vlfeat.torch",
   license = "MIT/X11",
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "builtin",
   modules = {
      ['vlfeat.init']           = 'init.lua',
      ['vlfeat.ffi']            = 'ffi.lua',
      ['vlfeat.kmeans']         = 'kmeans.lua',
   },
}
