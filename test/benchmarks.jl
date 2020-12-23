using DSP, BenchmarkTools, Statistics, DataFrames, CSV, Query

using DSP: _conv_os!, _conv_simple_fft!, _conv_direct!, _conv_direct_separable!,
    optimalfftfiltlength, nextfastfft

using DSP: _conv_direct!, _conv_direct_edge!, _conv_direct_core!, safeheadtail,
    safehead, safetail, _conv_center_range, dsp_reflect,
    _conv_direct_dim_edge_range, _conv_edge_b_offset, _conv_edge_e_offset,
    _conv_v_offset, stop_ndx, start_ndx, ConvOSThreadBuffers, _zeropad!,
    os_filter_transform!, _conv_os_core_kern_texec!

const BENCH_SECONDS = 2.5
const BENCH_MAX_NTRIALS = 10000

const MAX_B_SIZE = 16 * 2 ^ 30 # 16 GiB

nds = [1, 2, 3]
nus = [3, 5, 7, 10, 15, 20, 40, 60, 80, 100, 200, 400, 800, 1600, 10000, 50000, 100000]
nvs = nus[1:findlast(x -> x <= 10000, nus)]
nts = [1, 4]
elts = [Float64, Int]

to_size_tup(nu, ndims) = ntuple(i -> nu, ndims)

out_size(nu, nv, nd, v_vec) = ntuple(i -> v_vec && i > 1 ? nu : max(nu + nv - 1, 0), nd)

conv_rand(::Type{T}, sz) where T<:AbstractFloat = rand(T, sz)
conv_rand(::Type{T}, sz) where T<:Integer = rand(UnitRange{T}(-128, 128), sz)

function make_bases(::Type{T}, s::NTuple{N, Int}) where {T, N}
    bases = Vector{Array{T, N}}(undef, N)
    for i in 1:N
        sz = ntuple(j -> ifelse(j == i, s[j], 1), N)
        bases[i] = conv_rand(T, sz)
    end
    return bases
end

function allocate_test_arrays(nu, nv, nd, v_vec, elt::DataType = Float64)
    su = to_size_tup(nu, nd)
    sv = v_vec ? (nv,) : to_size_tup(nv, nd)
    u = conv_rand(elt, su)
    vbases = make_bases(elt, sv)
    v = conv_rand(elt, sv)
    sout = out_size(nu, nv, nd, v_vec)
    out = similar(u, sout)
    alt_out = similar(out)
    return out, alt_out, u, v, vbases
end

vec_to_arr(u) = reshape(u, (length(u), 1))

function bench_direct!(out, u, v, nt, args...)
    su, sv, sout = size(u), size(v), size(out)
    @benchmark(_conv_direct!($out, $u, $v, $su, $sv, $sout, nt = $nt),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_os!(out, u, v, nt)
    su, sv, sout = size(u), size(v), size(out)
    if eltype(out) <: Integer
        u, v, out = float(u), float(v), float(out)
    end
    nfft = map(optimalfftfiltlength, sv, su)
    @benchmark(_conv_os!($out, $u, $v, $su, $sv, $sout, $nfft, nt = $nt),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_simple_fft!(out, u, v, nt)
    su, sv, sout = size(u), size(v), size(out)
    if eltype(out) <: Integer
        u, v, out = float(u), float(v), float(out)
    end
    nfft = nextfastfft(sout)
    @benchmark(_conv_simple_fft!($out, $u, $v, $su, $sv, $sout, $nfft),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_separable_arrs!(out, alt_out, u, vs, nt)
    su, sout = size(u), size(out)
    svs = size.(vs)
    vec_mask = fill(false, size(vs))
    out_axes = axes(out)
    raw_input_range = axes(u)
    @benchmark(_conv_direct_separable!($out, $alt_out, $u, $vs, $su, $out_axes,
                                       $raw_input_range, $vec_mask, nt = $nt),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_separable_vecs!(out, alt_out, u, vs, nt)
    su, sout = size(u), size(out)
    svs = size.(vs)
    out_axes = axes(out)
    raw_input_range = axes(u)
    @benchmark(_conv_direct_separable!($out, $alt_out, $u, $vs, $su, $out_axes,
                                       $raw_input_range, nt = $nt),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_direct_not_small!(out, u, v, nt, args...)
    su, sv, sout = size(u), size(v), size(out)
    @benchmark(_conv_direct!($out, $u, $v, $su, $sv, $sout, nt = $nt,
                             use_small = false),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

function bench_direct_small!(out, u, v, nt, args...)
    su, sv, sout = size(u), size(v), size(out)
    @benchmark(_conv_direct!($out, $u, $v, $su, $sv, $sout, nt = $nt,
                             use_small = true),
               seconds = BENCH_SECONDS, samples = BENCH_MAX_NTRIALS)
end

bench_arr_vec_direct_small!(out, u, args...) =
    bench_direct_small!(out, vec_to_arr(u), args...)

bench_arr_vec_direct_not_small!(out, u, args...) =
    bench_direct_not_small!(out, vec_to_arr(u), args...)

function process_bench(t::BenchmarkTools.Trial)
    ts, m = t.times, t.memory
    ntrials = length(ts)
    median_t = median(ts)
    t_std = std(ts)
    return ntrials, median_t, t_std, m
end

function append_bench_trial!(io, df, elt, su, sv, nt, f, t)
    ntrials, median_t, t_std, m = process_bench(t)
    push!(df, (elt, su, sv, nt, Symbol(f), ntrials, median_t, t_std, m))
    CSV.write(io, df[end:end, :], delim = '\t', append = true)
end

function run_bench_point!(io, df, elt, su, sv, nt, f, args...)
    t = f(args...)
    append_bench_trial!(io, df, elt, su, sv, nt, f, t)
end

function run_array_benches!(io, df, out, alt_out, u, v, vbases, elt, nts)
    su, sv, sout = size(u), size(v), size(out)
    for nt in nts
        for f in (bench_direct!, bench_os!)
            @show f, nt
            run_bench_point!(io, df, elt, su, sv , nt, f, out, u, v, nt)
        end
        if ndims(u) > 1
            for f in (bench_separable_arrs!, bench_separable_vecs!)
                @show f, nt
                run_bench_point!(io, df, elt, su, sv , nt, f, out, alt_out, u, vbases, nt)
            end
        end
    end
    run_bench_point!(io, df, elt, su, sv , 1, bench_simple_fft!, out, u, v, 1)
end

function run_vec_benches!(io, df, out, u, v, vbases, elt, nts)
    su, sv, sout = size(u), size(v), size(out)
    fs = [bench_direct_not_small!,  bench_arr_vec_direct_not_small!]
    if sv[1] <= DSP.SMALL_FILT_CUTOFF
        append!(fs, [bench_direct_small!, bench_arr_vec_direct_small!])
    end
    for nt in nts
        for f in fs
            @show f, nt
            run_bench_point!(io, df, elt, su, sv , nt, f, out, u, v, nt)
        end
    end
end

function bench_vector!(io, df, out, alt_out, u, v, vbases, elt, nts)
    run_vec_benches!(io, df, out, u, v, vbases, elt, nts)
    u, v, out = expand_dims(u, v, out)
    run_array_benches!(io, df, out, alt_out, u, v, vbases, elt, nts)
end

function expand_dims(u, v, out)
    if ndims(u) != ndims(v)
        maxd = max(ndims(u), ndims(v))
        out = cat(out, dims = maxd)
        u = cat(u, dims = maxd)
        v = cat(v, dims = maxd)
    end
    u, v, out
end

alloc_size(nel::Int, elt) = nel * sizeof(elt)

alloc_size(sz::NTuple{<:Any, Int}, elt) = sizeof(elt) * prod(sz)

alloc_size(nel, nd, elt) = sizeof(elt) * nel ^ nd

alloc_size(nu, nv, nd, elt, v_vec) = alloc_size(nu, nd, elt) +
    alloc_size(nv, ifelse(v_vec, 1, nd), elt) +
    nd * alloc_size(nv, elt) +
    2 * alloc_size(out_size(nu, nv, nd, v_vec), elt)

function run_all_benches!(io, df, nds, nus, nvs, elts, nts; resume_pos = nothing)
    if resume_pos === nothing
        CSV.write(io, df[1:0, :], delim = '\t') # empty (write header)
        loop_nus = nus
        loop_nvs = nvs
    else
        upos = findfirst(isequal(resume_pos[1]), nus)
        loop_nus = nus[upos:end]
        loop_nvs = nvs[findfirst(isequal(resume_pos[2]), nvs):end]
    end
    for nu in loop_nus
        last_nv_ndx = findlast(x -> x <= nu, loop_nvs)
        last_nv_ndx === nothing && continue
        ok_nvs = loop_nvs[1:last_nv_ndx]
        for nv in ok_nvs
            for elt in elts
                # v X v
                nd = 1
                if alloc_size(nu, nv, nd, elt, false) <= MAX_B_SIZE
                    out, alt_out, u, v, vbases = allocate_test_arrays(
                        nu, nv, nd, false, elt
                    )
                    println("Bench-marking vector convolutions for:")
                    @show nu, nv, elt
                    bench_vector!(io, df, out, alt_out, u, v, vbases, elt, nts)
                end
                for nd in nds[2:end]
                    if alloc_size(nu, nv, nd, elt, true) <= MAX_B_SIZE
                        out, alt_out, u, v, vbases = allocate_test_arrays(
                            nu, nv, nd, true, elt
                        )
                        println("Bench-marking array-vector convolutions for:")
                        @show nu, nv, elt, nd
                        bench_vector!(io, df, out, alt_out, u, v, vbases, elt,
                                      nts)
                    end
                    if alloc_size(nu, nv, nd, elt, false) <= MAX_B_SIZE
                        println("Bench-marking array convolutions for:")
                        @show nu, nv, elt, nd
                        out, alt_out, u, v, vbases = allocate_test_arrays(
                            nu, nv, nd, false, elt
                        )
                        run_array_benches!(io, df, out, alt_out, u, v, vbases,
                                           elt, nts)
                    end
                end
            end
        end
        loop_nvs = nvs
    end
end

df = DataFrame(elt = DataType[], su = NTuple{<:Any, Int}[],
               sv = NTuple{<:Any, Int}[], nt = Int[], conv_method = Symbol[],
               ntrials = Vector{Int}(), median_t = Vector{Float64}(),
               t_std = Vector{Float64}(), mem = Int[])

fname = "/home/glynch/Downloads/bench_fast.csv"
dfraw = DataFrame(CSV.File(fname))
df_td = Dict("Float64" => Float64, "Int64" => Int)
to_ntuple = x -> eval(Meta.parse(x))
df = select(
    dfraw,
    :elt => ByRow(x -> df_td[x]) => :elt,
    :su => ByRow(to_ntuple) => :su,
    :sv => ByRow(to_ntuple) => :sv,
    :nt,
    :conv_method => ByRow(Symbol) => :conv_method,
    :nruns,
    :median_time,
    :time_std,
    :mem
)

x = @from i in df begin
    @where i.conv_method == :bench_direct! && i.nt == 1 && length(i.sv) == 2 && i.sv[1] == i.sv[2]
    @select {n = prod(i.su) * prod(i.sv), i.su, i.sv, i.median_time}
    @collect DataFrame
end


xf = @from i in df begin
    @where i.conv_method == :bench_direct! && i.nt == 1 && length(i.sv) == 2 && i.sv[1] == i.sv[2]  && i.elt == Float64
    @select {i.su, i.sv, n = prod(i.su) * prod(i.sv), n_core = prod(i.su .- i.sv .+ 1), i.median_time}
    @collect DataFrame
end

xff = @from i in df begin
    @where i.conv_method == :bench_direct! && i.nt == 1 && length(i.sv) == 2 && i.sv[1] == i.sv[2]  && i.elt == Float64
    @select {n = prod(i.su) * prod(i.sv), i.median_time}
    @collect DataFrame
end

xint = @from i in df begin
    @where i.conv_method == :bench_direct! && i.nt == 1 && length(i.sv) == 2 && i.sv[1] == i.sv[2] && i.elt == Int
    @select {n = prod(i.su) * prod(i.sv), i.median_time}
    @collect DataFrame
end


conv_edge_time(edge_nel, v_nel, ::Type{Float64}) = 0.307 * edge_nel * v_nel + 4528

# why not just use edge all the time if center is slower?
conv_center_time(center_nel, v_nel, nt, ::Type{Float64}) = 0.737 * center_nel * v_nel + 465

function conv_time(outsize, center_range, su, sv, nt, elt)
    center_nel = prod(length.(center_range))
    edge_nel = prod(outsize) - center_nel
    v_nel = prod(sv)
    conv_edge_time(edge_nel, v_nel, elt) + conv_center_time(center_nel, v_nel, nt, elt) + 195
end

function edge_total_bench(out, perimeter_range, u, rv,
                    input_range, voffsets, v_starts, v_stops,
                    edge_b_offsets, edge_e_offsets, all_dims,
                    center_range, edge_ranges)
    for n_edges in all_dims
        _conv_direct_edge!(out, perimeter_range, Val(n_edges), u, rv,
                           input_range, voffsets, v_starts, v_stops,
                           edge_b_offsets, edge_e_offsets, all_dims,
                           center_range, edge_ranges)
    end
end

function _conv_edge_kern!(out::AbstractArray{T, N}, u, rv, test_region, voffsets,
                          v_starts, v_stops, edge_b_offsets, edge_e_offsets) where {T, N}
    voffset, VOffset = voffsets
    v_start, V_Start = v_starts
    v_stop, V_Stop = v_stops
    edge_b_offset, Edge_B_offset = edge_b_offsets
    edge_e_offset, Edge_E_offset = edge_e_offsets
    zeroidx = zero(Edge_B_offset)
    for out_R in safetail(test_region)
        overlap_R_lower = max(zeroidx, Edge_B_offset - out_R) + V_Start
        overlap_R_upper = min(zeroidx, Edge_E_offset - out_R) + V_Stop
        overlap_R = overlap_R_lower : overlap_R_upper
        u_R_offset = out_R - VOffset
        for out_i in safehead(test_region)
            overlap_r_lower = max(0, edge_b_offset - out_i) + v_start
            overlap_r_upper = min(0, edge_e_offset - out_i) + v_stop
            overlap_r = overlap_r_lower : overlap_r_upper
            u_r_offset = out_i - voffset
            accum = zero(T)
            for accum_I in overlap_R
                for accum_i in overlap_r
                    accum = muladd(
                        u[accum_i + u_r_offset, accum_I + u_R_offset],
                        rv[accum_i, accum_I],
                        accum
                    )
                end
            end
            out[out_i, out_R] = accum
        end
    end
end

function conv_edge_center_op_time(su, sv, nd, v_vec, elt, nt)
    u = conv_rand(elt, su)
    v = conv_rand(elt, sv)
    sout = out_size(nu, nv, nd, v_vec)
    out = similar(u, sout)
    su, sv, sout = size(u), size(v), size(out)
    outsize, input_range, output_range, use_small, N, T =
        sout, axes(u), axes(out), nothing, nd, elt

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

    center_nel = prod(length.(center_range))
    edge_nel = prod(outsize) - center_nel
    v_nel = length(v)
    center_nops = center_nel * v_nel

    bc = @benchmark _conv_direct_core!(out, u, rv, voffsets, center_range, nt = nt)
    return median(bc.times) / center_nops
end
