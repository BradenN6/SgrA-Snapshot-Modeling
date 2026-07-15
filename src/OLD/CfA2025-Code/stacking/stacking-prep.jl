using JLD2
using DataFrames
using Comrade
using CSV

using Distributions
using VLBIImagePriors
using Pigeons
using Krang
using VLBISkyModels
using CairoMakie
using PairPlots
using FINUFFT
using Serialization
using HypercubeTransform
using IOCapture
using Suppressor

include("SgrAfits.jl")
include("models.jl")

#include("EHTModelStacker.jl/src/loadrose.jl")

function convert_chain(data_path, outdir)
    m = match(r"scan(\d{3})", data_path)

    if m !== nothing
        scan = parse(Int, m.captures[1])
        scan = lpad(string(scan), 3, '0')
        println("Extracted scan number: $scan")
    else
        println("No match found")
    end
    
    data = JLD2.load(data_path)
    chain = data["chain"]
    chain_tbl = DataFrame(chain.sky)
    #chain_tbl[!, "incl"] = (180/π) .* chain_tbl[!, "incl"]

    post_obj = samplerinfo(chain)[:post]
    mjd = post_obj.lpost.data[1].config.mjd
    rel_time = (post_obj.lpost.data[1].config[:Ti][1])
    time = mjd + rel_time/24.0

    # fill dummy value for logz - not used in stacking 
    dfsum = DataFrame(:scan => [scan], :time => [time], :mjd => [mjd], :rel_time => [rel_time], :logz => [-1e300])

    CSV.write(joinpath(outdir, "scan$scan-skychain-wmod.csv"), chain_tbl)
    CSV.write(joinpath(outdir, "scan$scan-dfsum.csv"), dfsum)

    # remove mass-to-distance ratio, which is fixed 5.03 μas
    select!(chain_tbl, Not(:mod))

    CSV.write(joinpath(outdir, "scan$scan-skychain.csv"), chain_tbl)

    println("CSV's written.")

    return chain_tbl, dfsum
end

"""
From SgrAfits main
"""

@show SnapshotModeling.skyKrang

println("Julia version: ", VERSION)

fov = 200.0  # μas
pix = 80 # alternatively, 128
# 2.5 μas/pixel should be sufficient for Sgr A*

model = "krang"  # or mG-Ring 

println("Model: $model.")

sampler = "pigeons"  # dynesty,pigeons
task = "analysis"  # sampling,analysis
# TODO load pigeons chain jld2
analysis_source = "chain"  # pt,chain

# if dynesty is used
dynamic = true
nlive_points = 5000
dlogz = 0.001

# if pigeons is used
cluster_config = true
mpi = true
n_tempering_levels = 20
n_threads = 24
n_rounds = 14

#println("Arguments: $args")
println("Field of View: $fov μas x $fov μas.")
println("Pixel Resolution: $pix x $pix.")
println("Sampler: $sampler.")
println("Number of Live Points for Dynesty: $nlive_points.")
println("dlogz for Dynesty: $dlogz.")

data_path = joinpath((@__DIR__), "snapshot-modeling/scan157_pigeons/scan157-posterior_chain.jld2")
mpi_run = joinpath((@__DIR__), "results/all/2025-07-22-14-28-36-oSw4GHoe")

snapshot_filepath = joinpath(@__DIR__, "data_comrade",
    "hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan157.jls")

""" 
Load Data
"""

#if isempty(args)
#    snapshot_filepath = joinpath(@__DIR__, "data_comrade",
#        "hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan157.jls")
#else
#    snapshot_filepath = args[1]
#end

#if task == "sampling"
if sampler == "dynesty"
    dvis, output_dir, snapshot_ID = SnapshotModeling.load_data(snapshot_filepath, "nlive-$(nlive_points)_dlogz-$(dlogz)_deltaMD_dynamic_4")
elseif sampler == "pigeons"
    dvis, output_dir, snapshot_ID = SnapshotModeling.load_data(snapshot_filepath, "pigeons")
end
#end

"""
Priors
"""

priorRing = (
    r = Uniform(μas2rad(10.0), μas2rad(30.0)),
    w = Uniform(μas2rad(1.0), μas2rad(10.0)),
    ma = (Uniform(0.0, 0.5), Uniform(0.0, 0.5)),
    mp = (Uniform(0, 2π), Uniform(0, 2π)),
    τ = Uniform(0.0, 1.0),
    ξτ= Uniform(0.0, π),
    f = Uniform(0.0, 1.0),
    σg = Uniform(μas2rad(1.0), μas2rad(100.0)),
    τg = Exponential(1.0),
    ξg = Uniform(0.0, 1π),
    xg = Uniform(-μas2rad(80.0), μas2rad(80.0)),
    yg = Uniform(-μas2rad(80.0), μas2rad(80.0))
)

# physically-motivated priors
ϵ = 0.01  # tolerance to exclude zero

priorKrang = (
    # SgrA* likely low spin, should be unconstrained
    a = MixtureModel([Uniform(-0.99, -ϵ), Uniform(ϵ, 0.99)], [0.5, 0.5]),
    pa = Uniform(0.0, 2 * π),  # This should go from 0 to 2π
    mod = DeltaDist(5.03),  # microas
    incl = Uniform(1 * π / 180, 89 * π / 180),  # θo in radians
    θs = Uniform(40 / 180 * π, π / 2),  # in radians, based on GRMHD opening angles
    p1 = Uniform(0.1, 10.0),
    p2 = Uniform(1.0, 10.0),
    rpeak = Uniform(1, 10),  # 1-18 prev
	χ = Uniform(-π, π),
    ι = Uniform(0.0, π/2),
    βv = Uniform(0.0, 0.99), # SgrA* likely slow velo
    spec = Uniform(-1.0, 5.0),  # σ
	η = Uniform(-π, π)
)

"""
Sky Model, Instrument Model, and Posterior
"""

if model == "krang"
    g, skym, intmodel, post = SnapshotModeling.setup_sky_model(fov, pix, dvis, skyKrang, priorKrang)
elseif model == "mG-Ring"
    g, skym, intmodel, post = SnapshotModeling.setup_sky_model(fov, pix, dvis, skyRing, priorRing)
end

"""
Interact with Posterior
"""
# Draw a random sample to interact with the posterior.
# Evaluate Unnormalized log posterior density.
s = prior_sample(post)
println("Unnormalized Log Posterior Density: $(logdensityof(post, s))")

# Plot Sample Sky Model
m = skymodel(post, s)
figskym = imageviz(intensitymap(m, skym.grid))  # grab the grid from skym
save(joinpath(@__DIR__, output_dir, "$snapshot_ID-skym_sample.png"), figskym)

println("Threads: $(Threads.nthreads())")


"""
Convert

"""

outdir = joinpath((@__DIR__), "csv-chains-grmhdtest3599")
mkpath(outdir)

chaindir = "chains-grmhdtest3599/"
files = readdir(chaindir)

jld2_files = filter(f -> endswith(f, ".jld2"), files)

data_paths = joinpath.(chaindir, jld2_files)

for data_path in data_paths
    data_path = joinpath((@__DIR__), data_path)
    chain_tbl, df_sum = convert_chain(data_path, outdir)
end

#data_path = joinpath((@__DIR__), "scan157-posterior_chain.jld2")
#chain_tbl, df_sum = convert_chain(data_path, outdir)



#=
data_path = joinpath((@__DIR__), "chains/scan009-posterior_chain.jld2")
data = JLD2.load(data_path)
chain = data["chain"]

# dummy variable for log evidence
# similar environment with everything loaded since posterior is saved in
# chain metadata

chain_tbl = DataFrame(chain.sky)
# save as csv

post_obj = samplerinfo(chain)[:post]

# time relative to MJD of obs

mjd = post_obj.lpost.data[1].config.mjd
rel_time = post_obj.lpost.data[1].config[:Ti][1]
time = mjd + rel_time

# make_hdf5_chain in res_tochainh5, write2h5
# load_chains for each snapshot
# sort in increasing time 
# loop over all, load, vector of dfs each elt a chain, a df each row time and log evidence
# fill with a single number
# call write2h5
# hdf5 table object
# load stacker 
# file containing priors
# delete mod column from chain delta dist, fixed
# DataFrames select
# flat - uniform
# lower and upper bound in same units
# angular params - von mises
# call main_krang
# julia main.jl command line args stackerh5 thing and prior information
# it checkpoints: -r to restart if dies
# new version with smaller stdeviation for persistent variables
# add a column to table - restrict n or y based on spread - percent of range; spin, inc 
# run with restrict, run without

# main saves a bunch of chain files to a directory, where h5 file of stacked chains is 
# save as csv for each checkpoint 
# specify num MCMC steps, output/save at point for batching; save to csv 
# filename_batch$1, $2
# get chain -- fitting for mean and std value across all snapshots
# load: DF with mu_paramname sigma_paramname, mean and std/dispersion
# load individually and vcat tables to final table 
# plot final parameter values
# each a normal or von mises distribution; mean mean value, std spread bw snapshots
# example mean image
# postprocess - re-raytrace image

# mean image
# numbers
# posteriors
# plot like previous paper

# grab mean values, raytrace an image; for a few samples - spread, mean of that
# mean_paramname and std_paramname for many samples
# grab columns, strip mu 

# chain 
# select the mu_columns - dataframe = df_mu (stripped)
# df[1, :] for row 
# NamedTuple(df[1, :]) Comrade expects named tuple 
# grab keys of row keys(df_mu) convert to strings, strip off mu, re-call 
# feed into skymodelposterior and sample element
# m = skymodel(post, chain/sample) to do raytracing
# img = intensitymap(m, grid)
# do for all elements in mu, take mean

# leave off stdeviation for now 
# can take stdev of mean images for certainty of mean structures

# df_mu - posterior samples of mean image
# can plot 
# spin - fixed - what think spin is 
# can plot posteriors for avg structure 
# people will overinterpret 

# mean chain for estimate of value 
# plot over time 

# standard deviation

#KDE Kernel Density Estimation
=#