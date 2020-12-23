# This file was formerly a part of Julia. License is MIT: https://julialang.org/license

import Base: trailingsize, tail
import LinearAlgebra.BLAS
using LinearAlgebra: dot

import Base.Threads.@spawn
import Base.OneTo

const SMALL_FILT_CUTOFF = 58

const NormalAxes{N} = NTuple{N, OneTo{Int}}

_zerosi(b,a,T) = zeros(promote_type(eltype(b), eltype(a), T), max(length(a), length(b))-1)

"""
    filt(b, a, x, [si])

Apply filter described by vectors `a` and `b` to vector `x`, with an optional initial filter
state vector `si` (defaults to zeros).
"""
function filt(b::Union{AbstractVector, Number}, a::Union{AbstractVector, Number},
              x::AbstractArray{T}, si::AbstractArray{S} = _zerosi(b,a,T)) where {T,S}
    filt!(Array{promote_type(eltype(b), eltype(a), T, S)}(undef, size(x)), b, a, x, si)
end

# in-place filtering: returns results in the out argument, which may shadow x
# (and does so by default)

"""
    filt!(out, b, a, x, [si])

Same as [`filt`](@ref) but writes the result into the `out` argument, which may
alias the input `x` to modify it in-place.
"""
function filt!(out::AbstractArray, b::Union{AbstractVector, Number},
               a::Union{AbstractVector, Number}, x::AbstractArray{T},
               si::AbstractArray{S,N} = _zerosi(b,a,T)) where {T,S,N}
    isempty(b) && throw(ArgumentError("filter vector b must be non-empty"))
    isempty(a) && throw(ArgumentError("filter vector a must be non-empty"))
    a[1] == 0  && throw(ArgumentError("filter vector a[1] must be nonzero"))
    if size(x) != size(out)
        throw(ArgumentError("output size $(size(out)) must match input size $(size(x))"))
    end

    as = length(a)
    bs = length(b)
    sz = max(as, bs)
    silen = sz - 1
    ncols = trailingsize(x,2)

    if size(si, 1) != silen
        throw(ArgumentError("initial state vector si must have max(length(a),length(b))-1 rows"))
    end
    if N > 1 && trailingsize(si,2) != ncols
        throw(ArgumentError("initial state vector si must be a vector or have the same number of columns as x"))
    end

    size(x,1) == 0 && return out
    sz == 1 && return mul!(out, x, b[1]/a[1]) # Simple scaling without memory

    # Filter coefficient normalization
    if a[1] != 1
        norml = a[1]
        a = a ./ norml
        b = b ./ norml
    end
    # Pad the coefficients with zeros if needed
    bs<sz   && (b = copyto!(zeros(eltype(b), sz), b))
    1<as<sz && (a = copyto!(zeros(eltype(a), sz), a))

    initial_si = si
    for col = 1:ncols
        # Reset the filter state
        si = initial_si[:, N > 1 ? col : 1]
        if as > 1
            _filt_iir!(out, b, a, x, si, col)
        elseif bs <= SMALL_FILT_CUTOFF
            _small_filt_fir!(out, b, x, si, col)
        else
            _filt_fir!(out, b, x, si, col, silen)
        end
    end
    return out
end

# Transposed direct form II
function _filt_iir!(out, b, a, x, si, col)
    silen = length(si)
    @inbounds for i=1:size(x, 1)
        xi = x[i,col]
        val = muladd(xi, b[1], si[1])
        for j=1:(silen-1)
            si[j] = muladd(val, -a[j+1], muladd(xi, b[j+1], si[j+1]))
        end
        si[silen] = muladd(xi, b[silen+1], -a[silen+1]*val)
        out[i,col] = val
    end
end

function _filt_fir_row_thread_exec!(out, b, x, si_init, sibuffs, col, silen,
                                    input_range, xa, usesmall, tno_override = 0)
    tno = tno_override > 0 ? tno_override : Threads.threadid()
    sibuff = sibuffs[tno]
    ib = first(input_range)
    target_warm_b = ib - silen
    warm_deficit = max(first(xa[1]) - target_warm_b, 0)
    warm_range = target_warm_b + warm_deficit : ib - 1
    if warm_deficit > 0
        si_offset = silen - warm_deficit + 1
        copyto!(sibuff, si_offset, si_init, si_offset, warm_deficit)
    end
    _filt_fir_warm!(b, x, sibuff, col, silen, warm_range)
    if usesmall
        _small_filt_fir!(out, b, x, sibuff, col, input_range)
    else
        _filt_fir!(out, b, x, sibuff, col, silen, input_range)
    end
end

# Transposed direct form II
function _filt_fir!(out, b, x, si, col,
                    silen = size(si, 1), input_range = 1:size(x, 1))
    @inbounds for i in input_range
        xi = x[i,col]
        out[i,col] = muladd(xi, b[1], si[1])
        for j=1:(silen-1)
            si[j] = muladd(xi, b[j+1], si[j+1])
        end
        si[silen] = b[silen+1]*xi
    end
end

function _filt_fir_warm!(b, x, si, col, silen, input_range)
    @inbounds for i in input_range
        xi = x[i,col]
        for j=1:(silen-1)
            si[j] = muladd(xi, b[j+1], si[j+1])
        end
        si[silen] = b[silen+1]*xi
    end
end

#
# filt implementation for FIR filters (faster than Base)
#

for n = 2:SMALL_FILT_CUTOFF
    silen = n-1
    si = [Symbol("si$i") for i = 1:silen]
    # Transposed direct form II
    @eval function _filt_fir!(out, b::NTuple{$n}, x, siarr, col, input_range, xstride, outstride)
        x_offset = (col - 1) * xstride
        o_offset = (col - 1) * outstride

        $(Expr(:block, [:(@inbounds $(si[i]) = siarr[$i]) for i = 1:silen]...))
        @inbounds for i in input_range
            xi = x[i+x_offset]
            out[i+o_offset] = muladd(xi, b[1], $(si[1]))
            $(Expr(:block, [:($(si[j]) = muladd(xi, b[$(j+1)], $(si[j+1]))) for j = 1:(silen-1)]...))
            $(si[silen]) = b[$(silen+1)]*xi
        end
        $(Expr(:block, [:(@inbounds siarr[$i] = $(si[i])) for i = 1:silen]...))
    end
end

# Convert array filter tap input to tuple for small-filtering
let chain = :(throw(ArgumentError("invalid tuple size")))
    for n = SMALL_FILT_CUTOFF:-1:2
        chain = quote
            if length(h) == $n
                _filt_fir!(
                    out,
                    ($([:(@inbounds(h[$i])) for i = 1:n]...),),
                    x,
                    si,
                    col,
                    input_range,
                    xstride,
                    outstride
                )
            else
                $chain
            end
        end
    end

    @eval function _small_filt_fir!(
        out::AbstractArray,
        h::AbstractVector,
        x::AbstractArray,
        si,
        col,
        input_range = 1:size(x, 1),
        xstride = size(x, 1),
        outstride = size(out, 1)
    )
        $chain
    end
end

"""
    deconv(b,a) -> c

Construct vector `c` such that `b = conv(a,c) + r`.
Equivalent to polynomial division.
"""
function deconv(b::StridedVector{T}, a::StridedVector{T}) where T
    lb = size(b,1)
    la = size(a,1)
    if lb < la
        return [zero(T)]
    end
    lx = lb-la+1
    x = zeros(T, lx)
    x[1] = 1
    filt(b, a, x)
end


"""
    _zeropad!(padded::AbstractVector,
              u::AbstractVector,
              padded_axes = axes(padded),
              data_dest::Tuple = (1,),
              data_region = CartesianIndices(u))

Place the portion of `u` specified by `data_region` into `padded`, starting at
location `data_dest`. Sets the rest of `padded` to zero. This will mutate
`padded`. `padded_axes` must correspond to the axes of `padded`.

"""
@inline function _zeropad!(
    padded::AbstractVector,
    u::AbstractVector,
    padded_axes = axes(padded),
    data_dest::Tuple = (first(padded_axes[1]),),
    data_region = CartesianIndices(u),
)
    datasize = length(data_region)
    # Use axes to accommodate arrays that do not start at index 1
    data_first_i = first(data_region)[1]
    dest_first_i = data_dest[1]
    copyto!(padded, dest_first_i, u, data_first_i, datasize)
    padded[first(padded_axes[1]):dest_first_i - 1] .= 0
    padded[dest_first_i + datasize : end] .= 0

    padded
end

@inline function _zeropad!(
    padded::AbstractArray,
    u::AbstractArray,
    padded_axes = axes(padded),
    data_dest::Tuple = first.(padded_axes),
    data_region = CartesianIndices(u),
)
    fill!(padded, zero(eltype(padded)))
    dest_axes = UnitRange.(data_dest, data_dest .+ size(data_region) .- 1)
    dest_region = CartesianIndices(dest_axes)
    copyto!(padded, dest_region, u, data_region)

    padded
end

"""
    _zeropad(u, padded_size, [data_dest, data_region])

Creates and returns a new base-1 index array of size `padded_size`, with the
section of `u` specified by `data_region` copied into the region of the new
 array as specified by `data_dest`. All other values will be initialized to
 zero.

If either `data_dest` or `data_region` is not specified, then the defaults
described in [`_zeropad!`](@ref) will be used.
"""
function _zeropad(u, padded_size, args...)
    padded = similar(u, padded_size)
    _zeropad!(padded, u, axes(padded), args...)
end

function _zeropad_keep_offset(u, padded_size, u_axes, args...)
    ax_starts = first.(u_axes)
    new_axes = UnitRange.(ax_starts, ax_starts .+ padded_size .- 1)
    _zeropad!(similar(u, new_axes), u, args...)
end

function _zeropad_keep_offset(
    u, padded_size, ::NormalAxes, args...
)
    _zeropad(u, padded_size, args...)
end

"""
    _zeropad_keep_offset(u, padded_size, [data_dest, dat_region])

Like [`_zeropad`](@ref), but retains the first index of `u` when creating a new
array.
"""
function _zeropad_keep_offset(u, padded_size, args...)
    _zeropad_keep_offset(u, padded_size, axes(u), args...)
end

"""
Estimate the number of floating point multiplications per output sample for an
overlap-save algorithm with fft size `nfft`, and filter size `nb`.
"""
os_fft_complexity(nfft, nb) =  (nfft * log2(nfft) + nfft) / (nfft - nb + 1)

"""
Determine the length of FFT that minimizes the number of multiplications per
output sample for an overlap-save convolution of vectors of size `nb` and `nx`.
"""
function optimalfftfiltlength(nb, nx)
    nfull = nb + nx - 1

    # Step through possible nffts and find the nfft that minimizes complexity
    # Assumes that complexity is convex
    first_pow2 = ceil(Int, log2(nb))
    max_pow2 = ceil(Int, log2(nfull))
    prev_complexity = os_fft_complexity(2 ^ first_pow2, nb)
    pow2 = first_pow2 + 1
    while pow2 <= max_pow2
        new_complexity = os_fft_complexity(2 ^ pow2, nb)
        new_complexity > prev_complexity && break
        prev_complexity = new_complexity
        pow2 += 1
    end
    nfft = pow2 > max_pow2 ? 2 ^ max_pow2 : 2 ^ (pow2 - 1)

    if nfft > nfull
        # If nfft > nfull, then it's better to find next fast power
        nfft = nextfastfft(nfull)
    end

    nfft
end

"""
Prepare buffers and FFTW plans for convolution. The two buffers, tdbuff and
fdbuff may be an alias of each other, because complex convolution only requires
one buffer. The plans are mutating where possible, and the inverse plan is
unnormalized.
"""
struct ConvOSThreadBuffers{T, A, B, P, I}
    tdbuff::A
    fdbuff::B
    p::P
    ip::I
end

function ConvOSThreadBuffers(
    ::Type{T}, tdbuff::A, fdbuff::B, p::P, ip::I
) where {T<:Real, A<:AbstractArray, B<:AbstractArray, P, I}
    ConvOSThreadBuffers{T, A, B, P, I}(tdbuff, fdbuff, p, ip)
end

function ConvOSThreadBuffers(
    ::Type{T}, buff::A, p::P, ip::I
) where {T<:Complex, A<:AbstractArray, P, I}
    ConvOSThreadBuffers{T, A, A, P, I}(buff, buff, p, ip)
end

function ConvOSThreadBuffers(u::AbstractArray{T, N},
                             nffts) where {T<:Real, N}
    tdbuff = similar(u, nffts)
    bufsize = ntuple(i -> i == 1 ? nffts[i] >> 1 + 1 : nffts[i], N)
    fdbuff = similar(u, Complex{T}, NTuple{N, Int}(bufsize))

    p = plan_rfft(tdbuff)
    ip = plan_brfft(fdbuff, nffts[1])

    ConvOSThreadBuffers(T, tdbuff, fdbuff, p, ip)
end

function ConvOSThreadBuffers(u::AbstractArray{T}, nffts) where T <: Complex
    buff = similar(u, nffts)
    p = plan_fft!(buff)
    ip = plan_bfft!(buff)
    ConvOSThreadBuffers(T, buff, p, ip)
end

"""
Transform the smaller convolution input to frequency domain, and return it in a
new array. However, the contents of `buff` may be modified.
"""
@inline function os_filter_transform!(buff::AbstractArray{<:Real}, p)
    p * buff
end

@inline function os_filter_transform!(buff::AbstractArray{<:Complex}, p!)
    copy(p! * buff) # p operates in place on buff
end

"""
Take a block of data, and convolve it with the smaller convolution input. This
may modify the contents of `tdbuff` and `fdbuff`, and the result will be in
`tdbuff`.
"""
@inline function os_conv_block!(buffs::ConvOSThreadBuffers{<:Real}, filter_fd)
    unsafe_execute!(buffs.p, buffs.tdbuff, buffs.fdbuff)
    buffs.fdbuff .*= filter_fd
    unsafe_execute!(buffs.ip, buffs.fdbuff, buffs.tdbuff)
end

"Like the real version, but only operates on one buffer"
@inline function os_conv_block!(buffs::ConvOSThreadBuffers{<:Complex}, filter_fd)
    unsafe_execute!(buffs.p, buffs.tdbuff, buffs.tdbuff)
    buffs.tdbuff .*= filter_fd
    unsafe_execute!(buffs.ip, buffs.tdbuff, buffs.tdbuff)
end

# Used by `unsafe_conv_kern_os!` to handle blocks of input data that need to be padded.
#
# For a given number of edge dimensions, convolve all regions along the
# perimeter that have that number of edge dimensions
#
# For a 3d cube, if n_edges = 1, this means all faces. If n_edges = 2, then
# all edges. Finally, if n_edges = 3, then all corners.
#
# This needs to be a separate function for subsets to generate tuple elements,
# which is only the case if the number of edges is a Val{n} type. Iterating over
# the number of edges with Val{n} is inherently type unstable, so this function
# boundary allows dispatch to make efficient code for each number of edge
# dimensions.
function _conv_os_edge!(
    # These arrays and buffers will be mutated
    out::AbstractArray{<:Any, N},
    os_thread_buff,
    perimeter_range,
    # Number of edge dimensions to pad and convolve
    n_edges::Val,
    # Data to be convolved
    u,
    filter_fd,
    # Sizes, ranges, and other pre-calculated constants
    #
    ## ranges listing center and edge blocks
    edge_ranges,
    center_block_ranges,
    ## size and axis information
    all_dims, # 1:N
    su,
    u_start,
    sv,
    nffts,
    out_start,
    out_stop,
    save_blocksize,
    sout_deficit, # How many output samples are missing if nffts > sout
    tdbuff_axes,
) where N
    # Iterate over all possible combinations of edge dimensions for a number of
    # edges
    #
    # For a 3d cube with n_edges = 1, this will specify the top and bottom faces
    # (edge_dims = (1,)), then the left and right faces (edge_dims = (2,)), then
    # the front and back faces (edge_dims = (3,))
    for edge_dims in subsets(all_dims, n_edges)
        # Specify a region on the perimeter by combining an edge block index for
        # each dimension on an edge, and the central blocks for dimensions not
        # on an edge.
        #
        # First make all entries equal to the center blocks:
        copyto!(perimeter_range, 1, center_block_ranges, 1, N)

        # For the dimensions chosen to be on an edge (edge_dims), get the
        # ranges of the blocks that would need to be padded (lie on an edge)
        # in that dimension.
        #
        # There can be one to two such ranges for each dimension, because with
        # some inputs sizes the whole convolution is just one range
        # (one edge block), or the padding will be needed at both the leading
        # and trailing side of that dimension
        selected_edge_ranges = getindex.(Ref(edge_ranges), edge_dims)

        # Visit each combination of edge ranges for the edge dimensions chosen.
        # For a 3d cube with n_edges = 1 and edge_dims = (1,), this will visit
        # the top face, and then the bottom face.
        for perimeter_edge_ranges in Iterators.ProductIterator(selected_edge_ranges)
            # The center region for non-edge dimensions has been specified above,
            # so finish specifying the region of the perimeter for this edge
            # block
            @inbounds for (i, dim) in enumerate(edge_dims)
                perimeter_range[dim] = perimeter_edge_ranges[i]
            end

            # Region of block indices, not data indices!
            block_region = CartesianIndices(
                NTuple{N, UnitRange{Int}}(perimeter_range)
            )
            @inbounds for block_pos in block_region
                # Figure out which portion of the input data should be transformed

                block_idx = convert(NTuple{N, Int}, block_pos)
                ## data_offset is NOT the beginning of the region that will be
                ## convolved, but is instead the beginning of the unaliased data.
                data_offset = save_blocksize .* (block_idx .- 1)
                ## How many zeros will need to be added before the data
                pad_before = max.(0, sv .- data_offset .- 1)
                data_ideal_stop = data_offset .+ save_blocksize
                ## How many zeros will need to be added after the data
                pad_after = max.(0, data_ideal_stop .- su)

                ## Data indices, not block indices
                data_region = CartesianIndices(
                    UnitRange.(
                        u_start .+ data_offset .- sv .+ pad_before .+ 1,
                        u_start .+ data_ideal_stop .- pad_after .- 1
                    )
                )

                # Convolve portion of input

                _zeropad!(os_thread_buff.tdbuff, u, tdbuff_axes, pad_before .+ 1, data_region)
                os_conv_block!(os_thread_buff, filter_fd)

                # Save convolved result to output

                block_out_stop = min.(
                    out_start .+ data_offset .+ save_blocksize .- 1,
                    out_stop
                )
                block_out_region = CartesianIndices(
                    UnitRange.(out_start .+ data_offset, block_out_stop)
                )
                ## If the input could not fill tdbuff, account for that before
                ## copying the convolution result to the output
                u_deficit = max.(0, pad_after .- sv .+ 1)
                valid_buff_region = CartesianIndices(
                    UnitRange.(sv, nffts .- u_deficit .- sout_deficit)
                )
                copyto!(out, block_out_region, os_thread_buff.tdbuff, valid_buff_region)
            end
        end
    end
end

# Assumes u is larger than, or the same size as, v
# nfft should be greater than or equal to 2*sv-1
function _conv_os!(out, u::AbstractArray{<:Any, N}, v, su, sv, sout, nffts;
                   nt = 1) where N
    u_start = first.(axes(u))
    out_axes = axes(out)
    out_start = first.(out_axes)
    out_stop = last.(out_axes)
    ideal_save_blocksize = nffts .- sv .+ 1
    # Number of samples that are "missing" if the output is smaller than the
    # valid portion of the convolution
    sout_deficit = max.(0, ideal_save_blocksize .- sout)
    # Size of the valid portion of the convolution result
    save_blocksize = ideal_save_blocksize .- sout_deficit
    nblocks = cld.(sout, save_blocksize)

    # Pre-allocation
    os_thread_buff = ConvOSThreadBuffers(u, nffts)
    tdbuff = os_thread_buff.tdbuff
    tdbuff_axes = axes(tdbuff)

    # Transform the smaller filter
    _zeropad!(tdbuff, v)
    filter_fd = os_filter_transform!(os_thread_buff.tdbuff, os_thread_buff.p)
    filter_fd .*= 1 / prod(nffts) # Normalize once for bfft

    # block indices for center blocks, which need no padding
    first_center_blocks = cld.(sv .- 1, save_blocksize) .+ 1
    last_center_blocks = fld.(su, save_blocksize)
    center_block_range = UnitRange.(first_center_blocks, last_center_blocks)

    # block index ranges for blocks that need to be padded
    # Corresponds to the leading and trailing side of a dimension, or if there
    # are no center blocks, corresponds to the whole dimension
    edge_ranges = map(nblocks, first_center_blocks, last_center_blocks) do nblock, firstfull, lastfull
        lastfull > 1 ? [1:firstfull - 1, lastfull + 1 : nblock] : [1:nblock]
    end
    all_dims = 1:N
    # Buffer to store ranges of indices for a single region of the perimeter
    perimeter_range = Vector{UnitRange{Int}}(undef, N)

    # Convolve all blocks that require padding.
    #
    # This is accomplished by dividing the perimeter of the volume into
    # subsections, where the division is done by the number of edge dimensions.
    # For a 3d cube, this convolves the perimeter in the following order:
    #
    # Number of Edge Dimensions | Convolved Region
    # --------------------------+-----------------
    #                         1 | Faces of Cube
    #                         2 | Edges of Cube
    #                         3 | Corners of Cube
    #
    for n_edges in all_dims
        _conv_os_edge!(
            # These arrays and buffers will be mutated
            out,
            os_thread_buff,
            perimeter_range,
            # Number of edge dimensions to pad and convolve
            Val(n_edges),
            # Data to be convolved
            u,
            filter_fd,
            # Sizes, ranges, and other pre-calculated constants
            #
            ## ranges listing center and edge blocks
            edge_ranges,
            center_block_range,
            ## size and axis information
            all_dims, # 1:N
            su,
            u_start,
            sv,
            nffts,
            out_start,
            out_stop,
            save_blocksize,
            sout_deficit,
            tdbuff_axes) # How many output samples are missing if nffts > sout
    end

    _conv_os_core!(
        out, os_thread_buff, filter_fd, u, sv, nffts, u_start, out_start,
        center_block_range, save_blocksize, nt
    )

    out
end

function _conv_os_core!(out, os_thread_buff, filter_fd, u, sv, nffts, u_start,
                        out_start, center_block_range, save_blocksize, nt)
    center_block_region = CartesianIndices(center_block_range)
    isempty(center_block_region) && return
    tdbuff_region = CartesianIndices(os_thread_buff.tdbuff)
    # Portion of buffer with valid result of convolution
    valid_buff_region = CartesianIndices(UnitRange.(sv, nffts))
    if nt == 1
        _conv_os_core_kern!(
            out, os_thread_buff, filter_fd, u, sv, nffts, u_start,
            out_start, center_block_region, save_blocksize, tdbuff_region,
            valid_buff_region
        )
    else
        nsysthr = Threads.nthreads()
        rsi = RegionSplitIter(center_block_region, nt)
        nchunk = length(rsi)
        tasks = Vector{Task}(undef, nchunk - 1)
        # Make buffers and fftw plans for each thread
        os_buffs = Vector{typeof(os_thread_buff)}(undef, nsysthr)
        os_buffs[Threads.threadid()] = os_thread_buff
        for tno in 1:nsysthr
            if !isassigned(os_buffs, tno)
                os_buffs[tno] = ConvOSThreadBuffers(u, nffts)
            end
        end
        for (cno, chunk_regions) in enumerate(rsi)
            if cno < nchunk
                tasks[cno] = @spawn(
                    _conv_os_core_kern_texec!(
                        out, os_buffs, filter_fd, u, sv, nffts, u_start,
                        out_start, chunk_regions, save_blocksize, tdbuff_region,
                        valid_buff_region
                    )
                )
            else
                _conv_os_core_kern_texec!(
                    out, os_buffs, filter_fd, u, sv, nffts, u_start,
                    out_start, chunk_regions, save_blocksize, tdbuff_region,
                    valid_buff_region
                )
            end
        end
        foreach(wait, tasks)
    end
end

function _conv_os_core_kern_texec!(out, os_thread_buffs, filter_fd, u, sv,
                                   nffts, u_start, out_start, chunk_regions,
                                   args...)
    buff = os_thread_buffs[Threads.threadid()]
    for chunk in chunk_regions
        _conv_os_core_kern!(out, buff, filter_fd, u, sv, nffts, u_start,
                            out_start, chunk, args...)
    end
end

function _conv_os_core_kern!(
    out, os_thread_buff, filter_fd, u, sv, nffts, u_start, out_start,
    input_block_region, save_blocksize, tdbuff_region, valid_buff_region
)
    @inbounds for block_pos in input_block_region
        # Calculate portion of data to transform
        block_idx = Tuple(block_pos)
        ## data_offset is NOT the beginning of the region that will be
        ## convolved, but is instead the beginning of the unaliased data.
        data_offset = save_blocksize .* (block_idx .- 1)
        data_stop = data_offset .+ save_blocksize
        data_region = CartesianIndices(
            UnitRange.(u_start .+ data_offset .- sv .+ 1, u_start .+ data_stop .- 1)
        )

        # Convolve this portion of the data

        copyto!(os_thread_buff.tdbuff, tdbuff_region, u, data_region)
        os_conv_block!(os_thread_buff, filter_fd)

        # Save convolved result to output

        block_out_region = CartesianIndices(
            UnitRange.(data_offset .+ out_start, data_stop .+ out_start .- 1)
        )
        copyto!(out, block_out_region, os_thread_buff.tdbuff, valid_buff_region)
    end
end

function _conv_simple_fft!(out, u::AbstractArray{T, N}, v::AbstractArray{T, N},
                         su, sv, outsize, nffts) where {T<:Real, N}
    padded = _zeropad(u, nffts)
    p = plan_rfft(padded)
    uf = p * padded
    _zeropad!(padded, v)
    vf = p * padded
    uf .*= vf
    raw_out = irfft(uf, nffts[1])
    copyto!(out, CartesianIndices(out), raw_out,
            CartesianIndices(UnitRange.(1, outsize)))
end

function _conv_simple_fft!(out, u, v, su, sv, outsize, nffts)
    upad = _zeropad(u, nffts)
    vpad = _zeropad(v, nffts)
    p! = plan_fft!(upad)
    p! * upad # Operates in place on upad
    p! * vpad
    upad .*= vpad
    ifft!(upad)
    copyto!(out, CartesianIndices(out), upad,
            CartesianIndices(UnitRange.(1, outsize)))
end

# v should be smaller than u for good performance
function _conv_fft!(out, u, v, su, sv, outsize; nt = 1)
    os_nffts = map(optimalfftfiltlength, sv, su)
    if any(os_nffts .< outsize)
        _conv_os!(out, u, v, su, sv, outsize, os_nffts, nt = nt)
    else
        nffts = nextfastfft(outsize)
        _conv_simple_fft!(out, u, v, su, sv, outsize, nffts)
    end
end

is_vec(sa::NTuple) = sum(x -> x > 1, sa) <= 1
is_vec(a::AbstractArray) = is_vec(size(a))

function vec_dim(sa)
    first_nonscalar = findfirst(x -> x > 1, sa)
    keepdim::Int = first_nonscalar == nothing ? 1 : first_nonscalar
    return keepdim
end

function squash_vec(a::AbstractArray{<:Any, N}, keepdim::Int) where N
    dims_to_drop = ntuple(i -> i + (i >= keepdim), N - 1)
    return dropdims(a, dims = dims_to_drop)
end

_conv_size(su::NTuple{N, Int}, sv::NTuple{N, Int}) where N = su .+ sv .- 1

function _conv_size(su::NTuple{N, Int}, sv::NTuple{1, Int}, dim = 1) where N
    ntuple(i -> i == dim ? su[i] + sv[i] - 1 : su[i], N)
end

function _conv_direct_separable_arrs!(out::AbstractArray{<:Any, N}, alt_out,
                                      arrs_alt_out, u, vs, su, svs, out_axes,
                                      input_range; kwargs...) where N
    nv = length(vs)
    nv == 0 && return input_range
    curr_input_range = input_range
    curr_si = su
    outsize = axes_to_size(out_axes)
    for i in 1:nv
        if iseven(nv - i  + arrs_alt_out)
            thisout = out
            inbuff = alt_out
        else
            thisout = alt_out
            inbuff = out
        end
        if i == 1
            thisin = u
        else
            thisin = inbuff
            curr_si = outsize
        end
        _conv_direct!(thisout, thisin, vs[i], curr_si, svs[i], outsize,
                      curr_input_range; kwargs...)
        filled_output = last.(curr_input_range) .+ svs[i] .- 1
        curr_input_range = OneTo.(filled_output)
    end
    return curr_input_range
end

function tuple_shiftndx(tupin::NTuple{N}, shiftndx::NTuple{N, Int}) where N
    ntuple(j -> tupin[shiftndx[j]], N)
end

axes_to_size(a) = last.(a) .- first.(a) .+ 1

function _conv_direct_separable_vecs!(out::AbstractArray{<:Any, N},
                                      alternate_out, u, vs, vs_dims, su,
                                      out_axes, input_range; kwargs...) where N
    nv = length(vs)
    nv == 0 && return out
    curr_input_range = input_range
    curr_dims = ntuple(identity, N)
    curr_outsize = axes_to_size(out_axes)
    curr_out = out
    curr_alt_out = alternate_out
    curr_input_range = input_range
    curr_si = su
    shifts = Vector{Int}(undef, N)
    for i in 1:nv
        vdim = vs_dims[i]
        v = vs[i]
        sv = size(v)
        if i == 1
            thisin = u
        else
            thisin = curr_alt_out
            curr_si = curr_outsize
        end
        if curr_dims[1] != vdim
            # permute array so its first dimension is the same as the vector's

            dim_pos::Int = findfirst(isequal(vdim), curr_dims)

            # calculating the shifts in the array before turning it into an
            # ntuple allows the compiler to infer the ntuple's type
            for j in 1:N
                @inbounds shifts[j] = mod(j + dim_pos - 2, N) + 1
            end
            shift_ndxs = ntuple(j -> @inbounds(shifts[j]), N)

            curr_dims = tuple_shiftndx(curr_dims, shift_ndxs)
            curr_outsize = tuple_shiftndx(curr_outsize, shift_ndxs)
            curr_si = curr_outsize
            perm_input_r = tuple_shiftndx(curr_input_range, shift_ndxs)
            curr_out = reshape(out, curr_outsize)
            permutedims!(curr_out, thisin, shift_ndxs)
            curr_alt_out = reshape(alternate_out, curr_outsize)
            curr_input_range = perm_input_r
            thisin = curr_out
        end
        _conv_direct!(curr_alt_out, thisin, v, curr_si, sv, curr_outsize,
                      curr_input_range; kwargs...)
        filled_output = _conv_size(last.(curr_input_range), sv)
        curr_input_range = OneTo.(filled_output)
    end
    if curr_dims != ntuple(identity, N)
        sp = sortperm(collect(curr_dims))
        shift_ndxs = ntuple(i -> sp[i], N)
        permutedims!(out, curr_alt_out, shift_ndxs)
    end
    return out
end

function default_vec_mask(svs, vec_mask)
    if vec_mask == nothing
        v_vec_mask = map(is_vec, svs)
    else
        if length(vec_mask) != length(svs)
            throw(ArgumentError("vec_mask must be `nothing` or the same length as `vs`"))
        end
        v_vec_mask = convert(BitVector, vec_mask)
    end
    v_vec_mask
end

# This does not work with offset arrays
function _conv_direct_separable!(out::AbstractArray, alt_out, u,
                                 vs::AbstractVector{<:AbstractArray}, su,
                                 out_axes::NormalAxes,
                                 input_range::NormalAxes = axes(u),
                                 vec_mask::Union{Nothing, AbstractVector{Bool}} = nothing;
                                 nt = 1,
                                 use_small = nothing)
    svs = size.(vs)
    v_vec_mask = default_vec_mask(svs, vec_mask)
    if any(v_vec_mask)
        v_vec_ndxs = findall(v_vec_mask)
        vvs_dims = map(vec_dim, svs[v_vec_ndxs])
        sp = sortperm(vvs_dims)
        vvs_dims .= vvs_dims[sp]

        arrs_alt_out = vvs_dims[1] != 1
        v_arr_ndxs = findall(.!v_vec_mask)
        curr_input_r = _conv_direct_separable_arrs!(out, alt_out, arrs_alt_out,
                                                    u, vs[v_arr_ndxs], su,
                                                    svs[v_arr_ndxs], out_axes,
                                                    input_range; nt = nt)

        vvs = squash_vec.(vs[v_vec_ndxs][sp], vvs_dims)
        _conv_direct_separable_vecs!(out, alt_out, u, vvs, vvs_dims, su, out_axes,
                                     curr_input_r; nt = nt, use_small = use_small)
    else
        arrs_alt_out = false
        curr_input_r = _conv_direct_separable_arrs!(out, alt_out, arrs_alt_out,
                                                    u, vs, su, svs, out_axes,
                                                    input_range; nt = nt)
    end
    return out
end

_out_offset(vr::OneTo{T}) where T = zero(T)
_out_offset(vr) = first(va)

function _conv_direct!(out::AbstractVector, u::AbstractVector, v::AbstractVector,
                       su, sv, outsize, input_range = 1:size(u, 1); nt = 1,
                       use_small = nothing)
    nv = sv[1]
    silen = nv - 1
    si = zeros(promote_type(eltype(u), eltype(v), eltype(out)), silen)
    if use_small === nothing
        use_small = nv <= SMALL_FILT_CUTOFF
    end
    if nt == 1
        if use_small
            _small_filt_fir!(out, v, u, si, 1, input_range)
        else
            _filt_fir!(out, v, u, si, 1, silen, input_range)
        end
    else
        ua = axes(u)
        chunks = splits(input_range, nt)
        nchunk = sum(!isempty, chunks)
        ntask = nchunk - 1
        tasks = Vector{Task}(undef, ntask)
        n_systhr = Threads.nthreads()
        sibuffs = Vector{typeof(si)}(undef, n_systhr  + 1)
        for i in 1:n_systhr
            sibuffs[i] = similar(si)
        end
        sibuffs[end] = si
        si_init = copy(si)
        for tno in 1:ntask
            tasks[tno] = @spawn _filt_fir_row_thread_exec!(
                out, v, u, si_init, sibuffs, 1, silen, chunks[tno], ua, use_small
            )
        end
        # Need the filter state, si, to be "correct" for the end of the signal
        _filt_fir_row_thread_exec!(
            out, v, u, si_init, sibuffs, 1, silen, chunks[nchunk], ua, use_small,
            n_systhr + 1
        )
        foreach(wait, tasks)
    end
    unsafe_copyto!(out, su[1] + 1, si, 1, silen)
end

function _conv_direct_cols_kern!(out, si, u, v, input_range, use_small,
                                 col_range, col_li, out_li, ur, silen)
    if use_small
        @inbounds for R in col_range
            colno = col_li[R]
            fill!(si, 0)
            _small_filt_fir!(out, v, u, si, colno, ur)
            outpos = out_li[last(input_range[1]) + 1, R]
            unsafe_copyto!(out, outpos, si, 1, silen)
        end
    else
        @inbounds for R in col_range
            fill!(si, 0)
            _filt_fir!(out, v, u, si, R, silen, ur)
            outpos = out_li[last(input_range[1]) + 1, R]
            unsafe_copyto!(out, outpos, si, 1, silen)
        end
    end
end

function _conv_direct_cols_texec!(out, sibuffs, u, v, input_range, use_small,
                                  col_ranges, col_li, out_li, ur, silen)
    si = sibuffs[Threads.threadid()]
    for col_range in col_ranges
        _conv_direct_cols_kern!(out, si, u, v, input_range, use_small, col_range,
                                col_li, out_li, ur, silen)
    end
end

function _conv_direct!(out::AbstractArray, u::AbstractArray, v::AbstractVector,
                       su, sv, outsize, input_range = axes(u); nt = 1,
                       use_small = nothing)
    nv = sv[1]
    silen = nv - 1
    if use_small === nothing
        use_small = nv <= SMALL_FILT_CUTOFF
    end

    sitype = promote_type(eltype(u), eltype(v), eltype(out))
    out_li = LinearIndices(out)
    col_li = LinearIndices(tail(su))
    input_region = CartesianIndices(input_range)
    col_range = safetail(input_region)
    row_range = safehead(input_region)
    if nt == 1
        si = Vector{sitype}(undef, silen)
        _conv_direct_cols_kern!(out, si, u, v, input_range, use_small, col_range,
                                col_li, out_li, row_range, silen)
    else
        rsi = RegionSplitIter(col_range, nt)
        nchunk = length(rsi)
        tasks = Vector{Task}(undef, nchunk - 1)
        nsysthr = Threads.nthreads()
        sibuffs = Vector{Vector{sitype}}(undef, nsysthr)
        for i in 1:nsysthr
            @inbounds sibuffs[i] = Vector{sitype}(undef, silen)
        end
        @inbounds for (i, chunks) in enumerate(rsi)
            if i < nchunk
                tasks[i] = @spawn(
                    _conv_direct_cols_texec!(out, sibuffs, u, v, input_range,
                                             use_small, chunks, col_li, out_li,
                                             row_range, silen)
                )
            else
                _conv_direct_cols_texec!(out, sibuffs, u, v, input_range,
                                         use_small, chunks, col_li, out_li,
                                         row_range, silen)
            end
        end
        foreach(wait, tasks)
    end
    out
end

_conv_center_range(ua::NormalAxes, va::NormalAxes) = UnitRange.(
    last.(va), last.(ua) .- first.(va) .+ 1
)
_conv_center_range(ua, va) = UnitRange.(
    first.(ua) .+ last.(va), last.(ua) .+ first.(va)
)

_conv_v_offset(va::NormalAxes) = CartesianIndex(last.(va))
_conv_v_offset(va) = CartesianIndex(first.(va) .+ last.(va))

_conv_edge_b_offset(ua, va) = CartesianIndex(first.(ua)) + CartesianIndex(last.(va))

_conv_edge_b_offset(ua::NormalAxes, va::NormalAxes) =
    CartesianIndex(last.(va))

_conv_edge_e_offset(ua, va) = CartesianIndex(last.(ua)) + CartesianIndex(first.(va))

_conv_edge_e_offset(ua::NormalAxes, va::NormalAxes) =
    CartesianIndex(last.(ua))

split_index_to_tuple((x, X)) = (x, Tuple(X))

start_ndx(ax) = CartesianIndex(first.(ax))
stop_ndx(ax) = CartesianIndex(last.(ax))

function _conv_direct_dim_edge_range(output_range::UnitRange{Int},
                                     center_range::UnitRange{Int})
    cstart, cstop = first(center_range), last(center_range)
    ostart, ostop = first(output_range), last(output_range)
    cstop >= cstart ? [ostart : cstart - 1, cstop + 1 : ostop] : [output_range]
end

_conv_direct_dim_edge_range(a::OneTo{Int}, b) where N =
    _conv_direct_dim_edge_range(convert(UnitRange{Int}, a), b)

_conv_direct_dim_edge_range(a, b::OneTo{Int}) =
    _conv_direct_dim_edge_range(a, convert(UnitRange{Int}, b))

function _conv_direct!(out::AbstractArray{T, N}, u::AbstractArray{<:Any, N},
                       v::AbstractArray{<:Any, N}, su, sv, outsize,
                       input_range = axes(u), output_range = axes(out); nt = 1,
                       use_small = nothing) where {T, N}
    va = axes(v)
    rv = dsp_reflect(v)
    center_range = _conv_center_range(input_range, va)
    perimeter_range = Vector{UnitRange{Int}}(undef, N)
    all_dims = 1:N
    edge_ranges = map(_conv_direct_dim_edge_range, output_range, center_range)
    voffsets = safeheadtail(_conv_v_offset(va))
    v_starts = safeheadtail(start_ndx(va))
    v_stops = safeheadtail(stop_ndx(va))
    edge_b_offsets = safeheadtail(_conv_edge_b_offset(input_range, va))
    edge_e_offsets = safeheadtail(_conv_edge_e_offset(input_range, va))
    for n_edges in all_dims
        _conv_direct_edge!(out, perimeter_range, Val(n_edges), u, rv,
                           voffsets, v_starts, v_stops,
                           edge_b_offsets, edge_e_offsets, all_dims,
                           center_range, edge_ranges)
    end
    _conv_direct_core!(out, u, rv, voffsets, center_range, nt = nt)
    out
end

function _conv_direct_edge!(out::AbstractArray{T, N}, perimeter_range,
                            n_edges::Val, u, rv, voffsets,
                            v_starts, v_stops, edge_b_offsets, edge_e_offsets,
                            all_dims, center_range, edge_ranges) where {T, N}
    voffset, VOffset = voffsets
    v_start, V_Start = v_starts
    v_stop, V_Stop = v_stops
    edge_b_offset, Edge_B_offset = edge_b_offsets
    edge_e_offset, Edge_E_offset = edge_e_offsets
    zeroidx = zero(Edge_B_offset)
    for edge_dims in subsets(all_dims, n_edges)
        copyto!(perimeter_range, 1, center_range, 1, N)
        selected_edge_ranges = getindex.(Ref(edge_ranges), edge_dims)
        for perimeter_edge_ranges in Iterators.ProductIterator(selected_edge_ranges)
            @inbounds for (i, dim) in enumerate(edge_dims)
                perimeter_range[dim] = perimeter_edge_ranges[i]
            end
            edge_region = CartesianIndices(
                NTuple{N, UnitRange{Int}}(perimeter_range)
            )
            for out_R in safetail(edge_region)
                overlap_R_lower = max(zeroidx, Edge_B_offset - out_R) + V_Start
                overlap_R_upper = min(zeroidx, Edge_E_offset - out_R) + V_Stop
                overlap_R = overlap_R_lower : overlap_R_upper
                u_R_offset = out_R - VOffset
                for out_i in safehead(edge_region)
                    overlap_r_lower = max(0, edge_b_offset - out_i) + v_start
                    overlap_r_upper = min(0, edge_e_offset - out_i) + v_stop
                    overlap_r = overlap_r_lower : overlap_r_upper
                    u_r_offset = out_i - voffset
                    accum = zero(T)
                    for accum_I in overlap_R
                        @inbounds for accum_i in overlap_r
                            accum = muladd(
                                u[accum_i + u_r_offset, accum_I + u_R_offset],
                                rv[accum_i, accum_I],
                                accum
                            )
                        end
                    end
                    @inbounds out[out_i, out_R] = accum
                end
            end
        end
    end
end

function _conv_direct_core!(
    out::AbstractArray, u, rv, voffsets, center_range; nt = 1
) where N
    center_region = CartesianIndices(center_range)
    isempty(center_region) && return
    voffset, VOffset = voffsets
    v_region = CartesianIndices(axes(rv))
    if nt == 1
        _conv_direct_core_kern!(out, u, rv, voffset, VOffset, center_region,
                                v_region)
    else
        rsi = RegionSplitIter(center_region, nt)
        nchunk = length(rsi)
        tasks = Vector{Task}(undef, nchunk - 1)
        for (cno, chunk_regions) in enumerate(rsi)
            if cno < nchunk
                tasks[cno] = @spawn(
                    _conv_direct_core_texec!(out, u, rv, voffset, VOffset,
                                             chunk_regions, v_region)
                )
            else
                _conv_direct_core_texec!(out, u, rv, voffset, VOffset,
                                         chunk_regions, v_region)
            end
        end
        foreach(wait, tasks)
    end
end

function _conv_direct_core_texec!(out, u, rv, voffset, VOffset, center_regions,
                                  v_region)
    for center_region in center_regions
        _conv_direct_core_kern!(out, u, rv, voffset, VOffset, center_region, v_region)
    end
end

function _conv_direct_core_kern!(out::AbstractArray{T}, u, rv, voffset, VOffset,
                                 center_region, v_region) where T
    vr, VR = safeheadtail(v_region)
    for OutPos in safetail(center_region)
        UOffset = OutPos - VOffset
        for outpos in safehead(center_region)
            uoffset = outpos - voffset
            accum = zero(T)
            @inbounds for J in VR, j in vr
                accum = muladd(u[uoffset+j, UOffset+J], rv[j,J], accum)
            end
            @inbounds out[outpos,OutPos] = accum
        end
    end
end

function _conv_similar(u, outsize, axesu, axesv)
    out_offsets = first.(axesu) .+ first.(axesv)
    out_axes = UnitRange.(out_offsets, out_offsets .+ outsize .- 1)
    similar(u, out_axes)
end

_conv_similar(u, outsize, ::NormalAxes, ::NormalAxes) = similar(u, outsize)

_conv_similar(u, v, outsize) = _conv_similar(u, outsize, axes(u), axes(v))

# Does convolution, will not switch argument order
function _conv!(out, u, v, su, sv, outsize)
    # TODO: Add spatial / time domain algorithm
    _conv_fft!(out, u, v, su, sv, outsize)
end

# Does convolution, will not switch argument order
function _conv(u, v, su, sv)
    outsize = su .+ sv .- 1
    out = _conv_similar(u, v, outsize)
    _conv!(out, u, v, su, sv, outsize)
end

# May switch argument order
"""
    conv(u,v)

Convolution of two arrays. Uses either FFT convolution or overlap-save,
depending on the size of the input. `u` and `v` can be  N-dimensional arrays,
with arbitrary indexing offsets, but their axes must be a `UnitRange`.
"""
function conv(u::AbstractArray{T, N},
              v::AbstractArray{T, N}) where {T<:BLAS.BlasFloat, N}
    su = size(u)
    sv = size(v)
    if prod(su) >= prod(sv)
        _conv(u, v, su, sv)
    else
        _conv(v, u, sv, su)
    end
end

function conv(u::AbstractArray{<:BLAS.BlasFloat, N},
              v::AbstractArray{<:BLAS.BlasFloat, N}) where N
    fu, fv = promote(u, v)
    conv(fu, fv)
end

conv(u::AbstractArray{<:Integer, N}, v::AbstractArray{<:Integer, N}) where {N} =
    round.(Int, conv(float(u), float(v)))

conv(u::AbstractArray{<:Number, N}, v::AbstractArray{<:Number, N}) where {N} =
    conv(float(u), float(v))

function conv(u::AbstractArray{<:Number, N},
              v::AbstractArray{<:BLAS.BlasFloat, N}) where N
    conv(float(u), v)
end

function conv(u::AbstractArray{<:BLAS.BlasFloat, N},
              v::AbstractArray{<:Number, N}) where N
    conv(u, float(v))
end

function conv(A::AbstractArray, B::AbstractArray)
    maxnd = max(ndims(A), ndims(B))
    return conv(cat(A, dims=maxnd), cat(B, dims=maxnd))
end

"""
    conv(u,v,A)

2-D convolution of the matrix `A` with the 2-D separable kernel generated by
the vectors `u` and `v`.
Uses 2-D FFT algorithm.
"""
function conv(u::AbstractVector{T}, v::AbstractVector{T}, A::AbstractMatrix{T}) where T
    # Arbitrary indexing offsets not implemented
    @assert !Base.has_offset_axes(u, v, A)
    m = length(u)+size(A,1)-1
    n = length(v)+size(A,2)-1
    B = zeros(T, m, n)
    B[1:size(A,1),1:size(A,2)] = A
    u = fft([u;zeros(T,m-length(u))])
    v = fft([v;zeros(T,n-length(v))])
    C = ifft(fft(B) .* (u * transpose(v)))
    if T <: Real
        return real(C)
    end
    return C
end


function check_padmode_kwarg(padmode::Symbol, su::Integer, sv::Integer)
    if padmode == :default_longest
        if su != sv
            Base.depwarn(
            """
            The default value of `padmode` will be changing from `:longest` to
            `:none` in a future release of DSP. In preparation for this change,
            leaving `padmode` unspecified is currently deprecated. To keep
            current behavior specify `padmode=:longest`. To avoid this warning,
            specify padmode = :none or padmode = :longest where appropriate.
            """
                ,
                :xcorr
            )
        end
        :longest
    else
        padmode
    end
end

function dsp_reverse(v::AbstractVector, ::NormalAxes)
    reverse(v, dims = 1)
end

function dsp_reverse(v::AbstractVector, vaxes)
    vsize = length(v)
    reflected_start = - first(vaxes[1]) - vsize + 1
    reflected_axes = (reflected_start : reflected_start + vsize - 1,)
    out = similar(v, reflected_axes)
    copyto!(out, reflected_start, Iterators.reverse(v), 1, vsize)
end

function dsp_reflect(v::AbstractArray) where N
    out = similar(v)
    v_region = CartesianIndices(v)
    IFirst = first(v_region)
    ILast = last(v_region)
    sv = size(v)
    @inbounds for offset in CartesianIndices(UnitRange.(0, sv .- 1))
        out[offset + IFirst] = v[ILast - offset]
    end
    out
end

"""
    xcorr(u,v; padmode = :longest)

Compute the cross-correlation of two vectors, by calculating the similarity
between `u` and `v` with various offsets of `v`. Delaying `u` relative to `v`
will shift the result to the right.

The size of the output depends on the padmode keyword argument: with padmode =
:none the length of the result will be length(u) + length(v) - 1, as with conv.
With padmode = :longest the shorter of the arguments will be padded so they are
equal length. This gives a result with length 2*max(length(u), length(v))-1,
with the zero-lag condition at the center.

!!! warning
    The default value of `padmode` will be changing from `:longest` to `:none`
    in a future release of DSP. In preparation for this change, leaving
    `padmode` unspecified is currently deprecated.
"""
function xcorr(
    u::AbstractVector, v::AbstractVector; padmode::Symbol = :default_longest
)
    su = size(u,1); sv = size(v,1)
    padmode = check_padmode_kwarg(padmode, su, sv)
    if padmode == :longest
        if su < sv
            u = _zeropad_keep_offset(u, sv)
        elseif sv < su
            v = _zeropad_keep_offset(v, su)
        end
        conv(u, dsp_reverse(conj(v), axes(v)))
    elseif padmode == :none
        conv(u, dsp_reverse(conj(v), axes(v)))
    else
        throw(ArgumentError("padmode keyword argument must be either :none or :longest"))
    end
end


struct RegionSplitIter{N}
    full_region::CartesianIndices{N, NTuple{N, UnitRange{Int}}}
    target_n_split::Int
    splitdim::Int
    trail_range::UnitRange{Int}

    function RegionSplitIter{N}(full_region::CartesianIndices{N, NTuple{N, UnitRange{Int}}},
                                target_n_split::Int, min_split_dim) where N
        min_split_dim <= N || throw(ArgumentError("min_split_dim must be less than or equal to N"))
        target_n_split < 0 && throw(ArgumentError("target_n_split must be non-negative"))
        raw_splitdim = range_splitdim(size(full_region), target_n_split)
        splitdim = max(min_split_dim, raw_splitdim)
        ntrail = 1
        for i in splitdim:N
            ntrail *= size(full_region, i)
        end
        new(full_region, target_n_split, splitdim, 1:ntrail)
    end
end

function RegionSplitIter(full_region::CartesianIndices{N, NTuple{N, UnitRange{Int}}},
                         target_n_split, min_split_dim = 0) where N
    RegionSplitIter{N}(full_region, target_n_split, min_split_dim)
end

function RegionSplitIter(full_region::CartesianIndices{N, NormalAxes{N}}, args...) where N
    RegionSplitIter(
        convert(CartesianIndices{N, NTuple{N, UnitRange{Int}}}, full_region),
        args...
    )
end

function iterate(iter::RegionSplitIter{N}, state = 1) where N
    if state > iter.target_n_split || isempty(iter.full_region)
        return nothing
    end
    r = split_range(state, iter.trail_range, iter.target_n_split)
    isempty(r) && return
    carts = range_to_cart_vec(iter.full_region, r, iter.splitdim)
    return carts, state + 1
end

IteratorSize(::Type{RegionSplitIter}) = HasLength()
IteratorEltype(::Type{RegionSplitIter}) = HasEltype()
eltype(iter::RegionSplitIter{N}) where N = Vector{CartesianIndices{N, NTuple{N, UnitRange{Int}}}}

function length(iter::RegionSplitIter{N}) where N
    len, rem = divrem(length(iter.trail_range), iter.target_n_split)
    thislen = len == 0 ? rem : iter.target_n_split
    return thislen
end

function range_splitdim(sz::NTuple{N, Int}, nsplit::Int) where N
    i = N
    p = 1
    while i > 0 && p < nsplit
        p *= sz[i]
        i -= 1
    end
    return i + 1
end

range_splitdim(::Tuple{}, ::Int) = 0

function range_to_cart_vec(c::C, r, splitdim) where C <: CartesianIndices
    sz = size(c)
    sd_sz = sz[splitdim]
    range_step = 1
    for i in 1 : splitdim - 1
        @inbounds range_step *= sz[i]
    end
    off = mod(first(r) - 1, sd_sz)
    ncart = cld(length(r) + off, sd_sz)
    carts = Vector{C}(undef, ncart)
    cart_nel = sd_sz - off
    ib = first(r)
    for i in 1:ncart
        ie = min(ib + cart_nel - 1, last(r))
        newcart = c[range_step * (ib - 1) + 1] : c[range_step * ie]
        @inbounds carts[i] = newcart
        ib = ie + 1
        cart_nel = sd_sz
    end
    return carts
end

function split_range(splitno, r, nsplit)
    nel = length(r)
    len, rem = divrem(nel, nsplit)
    if len == 0
        if splitno > rem
            rem = 0
        else
            len, rem = 1, 0
        end
    end
    f = first(r) + ((splitno-1) * len)
    l = f + len - 1
    if rem > 0
        if splitno <= rem
            f = f + (splitno - 1)
            l = l + splitno
        else
            f = f + rem
            l = l + rem
        end
    end
    return f:l
end

splits(r, nsplit) = map(x -> split_range(x, r, nsplit), 1:nsplit)

## Taken from ImageFiltering.jl
## Faster Cartesian iteration
# Splitting out the first dimension saves a branch
safetail(R::CartesianIndices) = CartesianIndices(tail(R.indices))
safetail(R::CartesianIndices{1}) = CartesianIndices(())
safetail(R::CartesianIndices{0}) = CartesianIndices(())
safetail(I::CartesianIndex) = CartesianIndex(tail(Tuple(I)))
safetail(::CartesianIndex{1}) = CartesianIndex(())
safetail(::CartesianIndex{0}) = CartesianIndex(())

safehead(R::CartesianIndices) = R.indices[1]
safehead(R::CartesianIndices{0}) = CartesianIndices(())
safehead(I::CartesianIndex) = I[1]
safehead(::CartesianIndex{0}) = CartesianIndex(())

safeheadtail(r) = (safehead(r), safetail(r))

function _conv_alg_estimate_runtime(
    ::Type{typeof(_conv_direct!)}, nthreads::Integer,
    arr_a_info::Tuple{Type{<:AbstractArray{T, N}}, NTuple{N, <:Integer}},
    arr_b_info::Tuple{Type{<:AbstractArray{S, M}}, NTuple{M, <:Integer}}
) where {T, N, S, M}
    _, su = arr_a_info
    _, sv = arr_b_info
    x = log(prod(su) * prod(sv))
    if nthreads == 1
        piecewise = x < 11 ?
            8.34283 + 0.20827 * x - 0.06407 * x^2 + 0.006047 * x^3 :
            -0.68556 + 1.04102 * x
    else
        center_size = ntuple(i -> sv[i] == 0 ? su[i] :
                             max(su[i] - sv[i] + 1, 0), N)
        using_single_t = all(center_size .== 1)
        if using_single_t || x > 17
            single_t_time = conv_alg_estimate_runtime(_conv_direct!, 1,
                                                      arr_a_info, arr_b_info)
            using_single_t && return single_t_time
            # Scale single t time
        else
            piecewise = 4.044 + 2.4908*x - 0.41892*x^2 + 0.028987*x^3 - 0.000606*x^4
        end
    end
    est = exp(piecewise)
end

function _conv_alg_estimate_runtime(
    ::Type{typeof(_conv_os!)}, nthreads::Integer,
    arr_a_info::Tuple{Type{<:AbstractArray{T, N}}, NTuple{N, <:Integer}},
    arr_b_info::Tuple{Type{<:AbstractArray{S, M}}, NTuple{M, <:Integer}}
) where {T, N, S, M}
    _, su = arr_a_info
    _, sv = arr_b_info
    nffts = optimalfftfiltlength.(sv, su)
    save_blocksize = nffts .- sv .+ 1
    outsize = su .+ sv .- 1
    nblocks = prod(cld.(outsize, save_blocksize))
    nfft_prod = prod(nffts)
    pred = nblocks * nfft_prod * log2(nfft_prod)
    x = log(pred)
    if x < 9.5
        piecewise = 10.23694074629848 + 0.06445546080160833*x
    elseif x < 13
        piecewise = 88.97279697607325 - 30.832984478857057*x +
            4.579218942191232*x^2 - 0.3002845288164446*x^3 +
            0.0073692177851378635*x^4
    else
        piecewise = -0.4509641362031153 + 1.0165204102745422*x
        # TODO: improve accuracy with large input, currently over-estimates by a factor of 2
    end
    if nt > 1 && x > 10.6
        if x > 15.5
            piecewise -= 1.177
        else
            piecewise -= 0.18
        end
    end
    est = exp(piecewise)
end

function _conv_alg_estimate_runtime(
    ::Type{typeof(_conv_simple_fft!)}, nthreads::Integer,
    arr_a_info::Tuple{Type{<:AbstractArray{T, N}}, NTuple{N, <:Integer}},
    arr_b_info::Tuple{Type{<:AbstractArray{S, M}}, NTuple{M, <:Integer}}
) where {T, N, S, M}
    _, su = arr_a_info
    _, sv = arr_b_info
    sout = su .+ sv .- 1
    nffts = nextfastfft.(sout)
    nel = prod(nffts)
    pred = nel * log2(nel)
    x = log(pred)
    if x < 7.85
        piecewise = 9.048406 + 0.677789*x - 0.10554*x^2 + 0.00585*x^3
    elseif x <= 11.82
        piecewise = 16.0928 - 1.26684*x + 0.103726*x^2 - 0.00185*x^3
    else
        piecewise = 2.030967 + 0.89380 * x
    end
    est = exp(piecewise)
end

