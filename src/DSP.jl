module DSP

using FFTW
using FFTW: unsafe_execute!
using LinearAlgebra: mul!, rmul!
using IterTools: subsets

import Base: iterate, IteratorSize, IteratorEltype, length, eltype

export conv, conv2, deconv, filt, filt!, xcorr

include("dspbase.jl")

include("util.jl")
include("unwrap.jl")
include("windows.jl")
include("periodograms.jl")
include("Filters/Filters.jl")
include("lpc.jl")
include("estimation.jl")

using Reexport
@reexport using .Util, .Windows, .Periodograms, .Filters, .LPC, .Unwrap, .Estimation

include("deprecated.jl")
end
