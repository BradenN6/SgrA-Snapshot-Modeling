using JLD2
using DataFrames
using Comrade
using CSV

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

    post_obj = samplerinfo(chain)[:post]
    mjd = post_obj.lpost.data[1].config.mjd
    rel_time = post_obj.lpost.data[1].config[:Ti][1]
    time = mjd + rel_time

    # fill dummy value for logz - not used in stacking 
    dfsum = DataFrame(:time => [time], :logz => [-1e300])

    CSV.write(joinpath(outdir, "scan$scan-skychain-wmod.csv"), chain_tbl)
    CSV.write(joinpath(outdir, "scan$scan-dfsum.csv"), dfsum)

    # remove mass-to-distance ratio, which is fixed 5.03 μas
    select!(chain_tbl, Not(:mod))

    CSV.write(joinpath(outdir, "scan$scan-skychain.csv"), chain_tbl)

    return chain_tbl, dfsum
end

outdir = joinpath((@__DIR__), "csv-chains")

chaindir = "chains"
files = readdir(chaindir)

jld2_files = filter(f -> endswith(f, ".jld2"), files)

data_paths = joinpath.(chaindir, jld2_files)

for data_path in data_paths
    data_path = joinpath((@__DIR__), data_path)
    chain_tbl, df_sum = convert_chain(data_path, outdir)
end



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