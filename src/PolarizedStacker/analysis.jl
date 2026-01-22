using Pkg; Pkg.activate("..")
Pkg.instantiate()
Pkg.precompile()

# TODO 20 opening angle

#using PythonCall
#using Printf
#using Tables
using CSV
using DataFrames
#using Comonicon
using EHTModelStacker
using Statistics
using Tables
#ehtim = pyimport("ehtim")

# Imports
using Comrade
using Distributions
using VLBIImagePriors, VLBISkyModels
#using Pigeons
using Krang
using VLBISkyModels

#using Dynesty. # TODO
using CairoMakie
using PairPlots
using FINUFFT
#using JLD2
#using Serialization
#using HypercubeTransform
#using IOCapture
#using Suppressor

include("models.jl")
include("SgrAfits.jl")

#include("stacker.jl")

function construct_meanstd(df, labels::Vector{String})
    means = DataFrame([Pair(l,df[!, "mean_"*l]) for l in labels])
    std = DataFrame([Pair(l, df[!, "stdd_"*l]) for l in labels])
    return means, std
end

function construct_meanstd(df)
    na = names(df)
    single = split.(na[findall(startswith("mean"), na)], "_") .|> last .|> String
    return construct_meanstd(df, single)
end


function calc_mean_std(df, column)
    column_mean = Statistics.mean(df[!, column])
    column_std = Statistics.std(df[!, column])
    column_median = Statistics.median(df[!, column])

    column_quantile = Statistics.quantile(df[!, column], [0.16, 0.5, 0.84])

    return column_mean, column_std, column_median, column_quantile
end


function mean_std_df(df::DataFrame)
    result = DataFrame()
    for name in names(df)
        col = df[!, name]
        # Skip non-numeric columns (all should be numeric)
        if eltype(col) <: Number
            m, s, med, quant = calc_mean_std(df, name)
            result[!, "$(name)-mean"] = [m]
            result[!, "$(name)-std"] = [s]
            result[!, "$(name)-median"] = [med]
            result[!, "$(name)-lowquantile"] = [quant[1]]
            result[!, "$(name)-medquantile"] = [quant[2]]
            result[!, "$(name)-upquantile"] = [quant[3]]
        end
    end
    return result
end


function image_reconstruction(post, chain, skym, output_dir; im_draws=false, fov=200.0, pix=128)
    """
    Reconstruct Images from sampled chain.
    """

    #theme = Theme(fontsize = 20)
    #set_theme!(theme)

    g = imagepixels(μas2rad(fov), μas2rad(fov), pix, pix; executor = ThreadsEx())

    # Image Reconstructions
    # Sample image from every 5 rows
    #samples = skymodel.(Ref(post), df[1:5:end, :])
    samples = skymodel.(Ref(post), chain[begin:3:end])
    #imgs = intensitymap.(samples, Ref(skym.grid))
    imgs = intensitymap.(samples, Ref(g))

    # Array of images
    mimg = Statistics.mean(imgs)
    simg = Statistics.std(imgs)
    figim = Figure(; resolution = (700, 700));
    axs = [Axis(figim[i, j], xreversed = true, aspect = 1) for i in 1:2, j in 1:2]
    image!(axs[1, 1], mimg, colormap = :afmhot); axs[1, 1].title = "Mean"
    image!(axs[1, 2], simg ./ (max.(mimg, 1.0e-8)), colorrange = (0.0, 2.0), colormap = :afmhot);axs[1, 2].title = "Std"
    image!(axs[2, 1], imgs[1], colormap = :afmhot);
    image!(axs[2, 2], imgs[end], colormap = :afmhot);
    hidedecorations!.(axs)
    save(joinpath(@__DIR__, output_dir, "image_reconstruction.png"), figim)

    # Plot Mean and Standard Deviation images
    figmeanimg = imageviz(mimg)
    save(joinpath(@__DIR__, output_dir, "meanimg.png"), figmeanimg)

    figstdimg = imageviz(simg)
    save(joinpath(@__DIR__, output_dir, "stdimg.png"), figstdimg)

    if im_draws
       # Image Samples
       mkpath(joinpath(@__DIR__, output_dir, "img_draws/"))

       for (idx, img) in enumerate(imgs[begin:10:end])
          figim = imageviz(img)  # grab the grid from skym
          save(joinpath(@__DIR__, output_dir, "img_draws/img-$idx.png"), figim)
       end
    end

    println("Image reconstruction complete.")

    return mimg, simg
end


function plot_posterior_density(fields, labels, titles, output_dir)
    """
    Plot Posterior Density of selected parameters/fields

    e.g. field = rad2μas(chain.sky.r) * 2, label = "Ring Diameter (μas)"
    """

    for (idx, field) in enumerate(fields)
        figpost = Figure(; resolution=(650, 400))
        p = density(figpost[1, 1], field, axis = (; xlabel = labels[idx], xlabelsize = 25, ylabelsize = 25))

        #ax = Axis(figpost[1, 1]; xlabel = labels[idx])
        #density!(ax, field)
        #figpost[1, 1] = ax
        save(joinpath(@__DIR__, output_dir, "postdens_$(titles[idx]).png"), figpost)
        println("Posterior $(titles[idx]) plotted.")
    end
end


function plot_sample_dists(means_df, std_df, field, prior, outdir)
    means_field = means_df[!, field]   
    std_field = std_df[!, field]

    means_plt = means_field[begin:200:end]
    std_plt = std_field[begin:200:end]
    
    lim = Statistics.mean(means_plt) + 5 * Statistics.mean(std_plt)

    # Define a range of x-values for plotting
    x = LinRange(-lim, lim, 1000)

    # Create a new figure and axis
    f = Figure()
    #ax = Axis(f[1, 1], xlabel="x", ylabel="pdf", title="$field Global Draws")
    ax = Axis(f[1, 1], title="Global Draws $field")

    min = minimum(prior[Symbol(field)])
    max = maximum(prior[Symbol(field)])

    # Plot each normal distribution
    for (μ, σ) in zip(means_plt, std_plt)
        d = truncated(Normal(μ, σ), min, max)
        d = Normal(μ, σ)
        lines!(ax, x, pdf.(d, x), label="μ=$(round(μ, digits=2)), σ=$(round(σ, digits=2))")
    end

    #axislegend(ax)
    
    save(joinpath(@__DIR__, outdir, "global_draws-$field.png"), f)
end

function plot_corners_stacked(chain, fields, title, output_dir, snapshot_ID)
    """
    Plot pair/corner plots of desired parameters

    Title to differentiate if multiple pairplots with different parameter
    subsets are desired.

    e.g. fields = [:a, :incl]

    """

    #theme = Theme(fontsize = 28)
    #set_theme!(theme)

    set_theme!(Theme(
        Axis = (
            xlabelsize = 28,
            ylabelsize = 28,
        )
    ))

    sky_chain = chain #.sky
    println("Sky Chain Parameters: $(propertynames(sky_chain))")
    println("Sky Chain Length: $(length(sky_chain))")
    println("Sky Chain Dimensions: $(ndims(sky_chain))")
    #subset = NamedTuple{Tuple(fields)}(Tuple((getproperty(sky_chain, f) for f in fields)))
    subset = NamedTuple{Tuple(fields)}(Tuple((getfield.(sky_chain, f) for f in fields)))
    pplt = pairplot(subset)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-pairplot_$title.png"), pplt)

    set_theme!()

    #set_theme!(Theme(
    #    Axis=(xlabelsize=20, ylabelsize=20, titlesize=22),
    #))

end


"""
SETUP
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

snapshot_filepath = joinpath(@__DIR__,
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
#figskym = imageviz(intensitymap(m, skym.grid))  # grab the grid from skym
#save(joinpath(@__DIR__, output_dir, "$snapshot_ID-skym_sample.png"), figskym)

#println("Threads: $(Threads.nthreads())")


"""
ANALYSIS

"""

outdir = joinpath((@__DIR__), "no-restrict")
mkpath(outdir)
df = CSV.read("$outdir/stacker_chain_ha_trunc-no-restrict.csv", DataFrame)
means, std = construct_meanstd(df)

CSV.write(joinpath(outdir, "means.csv"), means)
CSV.write(joinpath(outdir, "std.csv"), std)

mean_summary_df = mean_std_df(means)
std_summary_df = mean_std_df(std)
CSV.write(joinpath(outdir, "mean_summary.csv"), mean_summary_df)
CSV.write(joinpath(outdir, "std_summary.csv"), std_summary_df)

# Image reconstruction
# Desired column order
ordered_cols = ["a", "pa", "mod", "incl", "θs", "p1", "p2", "rpeak", "χ", "ι", "βv", "spec", "η"]

means_wmod = copy(means)

# Insert "mod" column with constant value (if not already in means_df)
means_wmod[!, "mod"] = fill(5.03, nrow(means_wmod))

# Reorder columns for image
available_ordered_cols = filter(col -> col in names(means_wmod), ordered_cols)
means_wmod = means_wmod[:, available_ordered_cols]

CSV.write(joinpath(outdir, "means-wmod.csv"), means_wmod)

means_wmod_acutpos = means_wmod[ (means_wmod.a .>= 0.0), :]

std_wmod = copy(std)

# Insert "mod" column with constant value (if not already in means_df)
std_wmod[!, "mod"] = fill(5.03, nrow(std_wmod))

# Reorder columns for image
available_ordered_cols = filter(col -> col in names(std_wmod), ordered_cols)
std_wmod = std_wmod[:, available_ordered_cols]

CSV.write(joinpath(outdir, "std-wmod.csv"), std_wmod)

# Tables.jl
chain = Tables.rowtable(means_wmod)
chain2 = map(chain) do row 
    (;sky = row,)
end

fields = [
    means_wmod[!, "a"],
    means_wmod[!, "pa"] * 180/π,
    means_wmod[!, "incl"] * 180/π,
    means_wmod[!, "θs"] * 180/π,
    means_wmod[!, "p1"],
    means_wmod[!, "p2"],
    means_wmod[!, "rpeak"],
    means_wmod[!, "χ"] * 180/π,
    means_wmod[!, "ι"] * 180/π,
    means_wmod[!, "βv"],
    means_wmod[!, "spec"],
    means_wmod[!, "η"] * 180/π
]
fields_std = [
    std_wmod[!, "a"],
    std_wmod[!, "pa"] * 180/π,
    std_wmod[!, "incl"] * 180/π,
    std_wmod[!, "θs"] * 180/π,
    std_wmod[!, "p1"],
    std_wmod[!, "p2"],
    std_wmod[!, "rpeak"],
    std_wmod[!, "χ"] * 180/π,
    std_wmod[!, "ι"] * 180/π,
    std_wmod[!, "βv"],
    std_wmod[!, "spec"],
    std_wmod[!, "η"] * 180/π
]
fields_acutpos = [
    means_wmod_acutpos[!, "a"],
    means_wmod_acutpos[!, "pa"] * 180/π,
    means_wmod_acutpos[!, "incl"] * 180/π,
    means_wmod_acutpos[!, "θs"] * 180/π,
    means_wmod_acutpos[!, "p1"],
    means_wmod_acutpos[!, "p2"],
    means_wmod_acutpos[!, "rpeak"],
    means_wmod_acutpos[!, "χ"] * 180/π,
    means_wmod_acutpos[!, "ι"] * 180/π,
    means_wmod_acutpos[!, "βv"],
    means_wmod_acutpos[!, "spec"],
    means_wmod_acutpos[!, "η"] * 180/π
]
#fields = [
#    chain2.sky.a,
#    chain2.sky.pa * 180/π,
#    #chain.sky.mod,
#    chain2.sky.incl * 180/π,
#    chain2.sky.θs * 180/π,
#    chain2.sky.p1,
#    chain2.sky.p2,
#    chain2.sky.rpeak,
#    chain2.sky.χ * 180/π,
#    chain2.sky.ι * 180/π,
#    chain2.sky.βv,
#    chain2.sky.spec,
#    chain2.sky.η * 180/π
#]
labels = [
    "Spin Parameter a",
    "Position Angle of Projected Spin Axis [Deg]",
    "Inclination Angle θo [Deg]",
    "Opening Angle θs [Deg]",
    "p1",
    "p2",
    "Rpeak",
    "χ [Deg]",
    "ι [Deg]",
    "βv [c]",
    "spec",
    "η [Deg]"
]
titles = [
    "a",
    "pa",
    "incl",
    "θs",
    "p1",
    "p2",
    "rpeak",
    "χ",
    "ι",
    "βv",
    "spec",
    "η"
]

#outdir = joinpath((@__DIR__), "analysis-results/")
#mkpath(outdir)

mkpath(outdir * "/mean")
mkpath(outdir * "/std")
mkpath(outdir * "/mean_acutpos")
mkpath(outdir * "/images")

plot_posterior_density(fields, labels, titles, outdir * "/mean")
plot_posterior_density(fields_std, labels, titles, outdir * "/std")
plot_posterior_density(fields_acutpos, labels, titles, outdir * "/mean_acutpos")

plt = lines(means.a)
save(joinpath(outdir, "lines.png"), plt)
# TODO use skym based on priors or just raytrace images with values through krang directly
image_reconstruction(post, chain2, skym, outdir * "/images"; im_draws=true, fov=200.0, pix=128)

# exponential prior

# TODO plot standard deviation density - chi measured well in each, but high std 
# mean chi hard to measure, noisy sample
# movie of posteriors and Gaussians
# table 

# measure of variability not std 
# average of sigma is average variability value, sigma is uncertainty on variability 

# avg samples from mean posterior 


# stdimg - uncertain near horizon 
# semianalytic

fields_dist = ["a", "pa", "incl", "θs", "p1", "p2", "rpeak", "χ", "ι", "βv", "spec", "η"]
for field in fields_dist
    plot_sample_dists(means, std, field, priorKrang, outdir * "/images")
end


pair_fields = (:a, :pa, :incl, :θs, :p1, :p2,
        :rpeak, :χ, :ι, :βv, :spec, :η) # no :mod

pair_fields_trunc = (:a, :pa, :incl, :θs,
    :rpeak, :χ)

plot_corners_stacked(chain, pair_fields, "all", outdir, snapshot_ID)
plot_corners_stacked(chain, pair_fields_trunc, "trunc", outdir, snapshot_ID)


# PLOT GAUSSIANS
# TODO truncated Gaussians
# TABLE of numbers
# TODO truncate priors 