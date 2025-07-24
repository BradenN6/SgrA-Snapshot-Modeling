using Pkg

Pkg.activate(@__DIR__)

println("Running with $(Threads.nthreads()) threads.")
println("Hostname: $(gethostname())")
println("CWD: $(pwd())")
#println("ENV: JULIA_NUM_THREADS=$(ENV["JULIA_NUM_THREADS"])")

using VLBISkyModels
using Distributions
using Comrade

module SnapshotModeling

# Imports
using Comrade
using Distributions
using VLBIImagePriors, VLBISkyModels
using Pigeons
using Krang
using VLBISkyModels

#using Dynesty. # TODO
using CairoMakie
using PairPlots
using FINUFFT
using JLD2
using Serialization
using HypercubeTransform
using IOCapture
using Suppressor

# using Pyehtim

#using DataFrames
#using StatsBase
#using Enzyme
#using DynamicPPL
#using Measurements

model_path = joinpath((@__DIR__), "models.jl")
include(model_path)

"""
Snapshot Modeling of Sgr A*

Author: Braden Nowicki with Paul Tiede
Last Updated: 2025-07-08
"""

#=
function convert_to_com(file, out, dataproduct)
    obs = ehtim.obsdata.load_uvfits(file)
    vis = extract_table(obs, dataproduct)
    serialize(out, vis)
end

function data_convert()
    """
    Serialize uvfits data

    Pyehtim must be loaded
    """

    files = filter(endswith(".uvfits"), readdir("noisefrac0.02", join=true))

    mkpath("data_comrade")

    outs = joinpath.("data_comrade", replace.(basename.(files), ".uvfits" => ".jls"))

    convert_to_com.(files, outs, Ref(Visibilities()))

    # Now if Comrade is loaded we can load the data
    # (you do not need Pyehtim loaded for this)
    deserialize(outs[3])
end
=#

function load_data(snapshot_filepath, addendum)
    """
    Load EHT Observations: Complex Visibilities

    Data is loaded by deserializing previously serialized uvfits files.
    This allows the data to be read without Python dependencies in Julia's
    native Conda management.

    To do so, uncommon convert_to_com() and data_convert() and run with
    Pyehtim loaded.
    """

    dvis = deserialize(snapshot_filepath)

    snapshot = collect(eachsplit(snapshot_filepath, "/"))[end]
    snapshot_ID = collect(eachsplit(collect(eachsplit(snapshot, "."))[end-1], "_"))[end]

    println("Snapshot Filepath: $snapshot_filepath")
    println("Snapshot: $snapshot")
    println("Snapshot ID: $snapshot_ID")

    # Extract the number
    scan_num = parse(Int, replace(snapshot_ID, "scan" => ""))

    # Pad the number with zeros using lpad()
    scan_num = lpad(string(scan_num), 3, '0')

    # Combine it back with "scan"
    snapshot_ID = "scan" * scan_num

    output_dir = "snapshot-modeling/" * snapshot_ID * "_" * addendum * "/"
    println("Output Directory: $output_dir")

    mkpath(joinpath(@__DIR__, output_dir))

    println("Data initialized.")

    return dvis, output_dir, snapshot_ID
end


function setup_sky_model(fov, pix, dvis, sky, prior; dynesty=false)
    """
    Set up the grid, sky model, and posterior. Optionally thread if available.

    200.0 μas to avoid clipping/hard edges, which cause ringing in Fourier
    domain.
    Sgr A* data is more sparse, less resolution required.
    Use FINUFFT for the Fourier transform and set threads since it doesn't benefit Sgr A*
    """

    g = imagepixels(μas2rad(fov), μas2rad(fov), pix, pix; executor = ThreadsEx())

    skym = SkyModel(sky, prior, g; algorithm=FINUFFTAlg(;threads=1))

    # Model instrument gains to use complex visibilities
    # Using IntegSeg() instead of ScanSeg() because the snapshots are short
    # (not full scans)
    G = SingleStokesGain() do x
        lg = x.lg
        gp = x.gp
        return exp(lg + 1im * gp)
    end

    if dynesty
        # For Dynesty fits to work, using Uniform phase distribution.
        # Not ideal, but von Mises priors have not been incorporated into
        # ascube() which dynesty requires.
        intpr = (
            lg = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
            gp = ArrayPrior(IIDSitePrior(IntegSeg(), Uniform(0.0, 2π)); refant = SEFDReference(0.0), phase = true),
        )
        
    else
        intpr = (
            lg = ArrayPrior(IIDSitePrior(IntegSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(IntegSeg(), Normal(0.0, 1.0))),
            gp = ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant = SEFDReference(0.0), phase = true),
        )
    end

    intmodel = InstrumentModel(G, intpr)

    # Posterior with instrument model fitting
    post = VLBIPosterior(skym, intmodel, dvis)

    println("Sky Model, Instrument Model, and Posterior initialized.")

    return g, skym, intmodel, post
end


function pigeons_sample(post, output_dir, snapshot_ID; mpi=true,
        n_tempering_levels=20, n_threads=24, n_rounds=14)
    """
    Sampling the Posteriors
    """

    # Flatten parameter space and move from constrained parameters to
    # (-∞, ∞) support using 'asflat'
    fpost = asflat(post)
    ndim = dimension(fpost)
    println("Flat Space Dimensions: $ndim")

    println("Threads: $(Threads.nthreads())")

    if mpi
        setup_mpi(
            submission_system = :slurm,
            environment_modules = ["intel", "intelmpi"], # gcc, openmpi for anvil
            mpiexec = """srun -n \$SLURM_NTASKS --mpi=pmi2 --mem-per-cpu 2G"""
        )

        mpi_run = pigeons(
            target = fpost,
            explorer = SliceSampler(),
            record = [traces, round_trip, Pigeons.timing_extrema, log_sum_ratio,
                        disk],
            multithreaded = false, # can be skipped, the default
            checkpoint = true,
            n_chains = n_tempering_levels,
            on = Pigeons.ChildProcess(
                    n_local_mpi_processes = n_tempering_levels,  # SBATCH --ntasks
                    n_threads = n_threads,
                    mpiexec_args = `--mpi=pmi2`
            ),
            n_rounds = n_rounds
        )

        pt = Pigeons.load(mpi_run)
    else
        # Multithreaded Pigeons
        pt = pigeons(target=fpost, explorer=SliceSampler(), 
            record=[traces, round_trip, log_sum_ratio, disk], n_chains=20, n_rounds=10, 
            multithreaded=true, checkpoint=true)
    end
    
    @save joinpath(@__DIR__, output_dir, "$snapshot_ID-pt.jld2") pt

    # Transforms back into parameter space with 'NamedTuple' format
    # Posterior chain matches units in prior
    chain = sample_array(fpost, pt)

    # Save transformed chain
    @save joinpath(@__DIR__, output_dir, "$snapshot_ID-posterior_chain.jld2") chain

    println("Pigeons Sampler Complete. Chain Saved.")
    println("See results/latest for checkpoints.")

    return fpost, pt, chain
end


function parse_slurm_output(output::String)
    lines = split(output, '\n')

    complete = false
    
    if length(lines) < 2
        println("No job data found.")
        complete = true
        return complete, nothing
    end

    # Try to parse the second line (first is header)
    second_line = strip(lines[2])
    if isempty(second_line)
        println("Second line is empty.")
        complete = true
        return complete, nothing
    end

    fields = split(second_line)
    job_id = fields[1]

    println("Job ID: $job_id")
    return complete, job_id
end



function pigeons_cluster(post, output_dir, snapshot_ID; mpi=true,
        n_tempering_levels=20, n_threads=24, n_rounds=14)
    """
    Run pigeons with the Cluster paradigm
    Pigeons creates its own submission script and handles MPI
    Run from the REPL (not a job script) and do not run analysis afterwards.
    """

    # Flatten parameter space and move from constrained parameters to
    # (-∞, ∞) support using 'asflat'
    fpost = asflat(post)
    ndim = dimension(fpost)
    println("Flat Space Dimensions: $ndim")

    println("Threads: $(Threads.nthreads())")

    println("Preamble Executed.")
    println("Snapshot: $snapshot_ID")

    #using Pigeons

    MPI_OUTPUT_PATH = joinpath((@__DIR__), snapshot_ID)
    SLURM_SUBMIT_DIR = @__DIR__
    preamble_path = joinpath(@__DIR__, "SgrAfits.jl")

    setup_mpi(
        submission_system = :slurm,
        environment_modules = ["intel", "intelmpi"],
        add_to_submission = ["#SBATCH -p blackhole"],
        mpiexec = """srun -n \$SLURM_NTASKS --mpi=pmi2 --mem-per-cpu 2G"""
    )

    mpi_run = Pigeons.pigeons(
        target = fpost,
        explorer = SliceSampler(),
        record = [traces, round_trip, Pigeons.timing_extrema, log_sum_ratio,
                    disk],
        #multithreaded = true,
        checkpoint = true,
        n_chains = n_tempering_levels,
        on = Pigeons.MPIProcesses(
            n_mpi_processes = n_tempering_levels,  # SBATCH ---ntasks
            walltime = "10-00:00:00",  # SBATCH -t
            n_threads = n_threads,  # 48 threads per node; don't want unused threads; #SBATCH --cpus-per-task
            memory = "2gb",  # #SBATCH --mem-per-cpu
            dependencies = [
                Pigeons, # <- Pigeons itself can be skipped, added automatically
                preamble_path,
            ],
        ),
        n_rounds = n_rounds
    )

    c = IOCapture.capture() do
        Pigeons.queue_status(mpi_run)
    end
    
    captured_output = c.output
    
    println("Output: $captured_output.")

    complete, job_id = parse_slurm_output(captured_output)

    println(@capture_out run(`squeue -p blackhole`))

    # wait until job is complete
    loop = true

    if complete == false
        while loop
            if contains((@capture_out run(`squeue -p blackhole`)), String(job_id))
                println("Sampling is ongoing. Sleeping for one hour.")
                sleep(3600)  # 1h in seconds 
            else
                loop = false
            end
        end
    end

    if !isnothing(job_id)
        println("Pigeons Sampling Job $job_id finished.")
    else
        println("Pigeons Sampling Job not submitted.")
    end

    pt = Pigeons.load(mpi_run)

    # Save
    @save joinpath(@__DIR__, output_dir, "$snapshot_ID-pt.jld2") pt

    # Transforms back into parameter space with 'NamedTuple' format
    # Posterior chain matches units in prior
    chain = sample_array(fpost, pt)

    # Save transformed chain
    @save joinpath(@__DIR__, output_dir, "$snapshot_ID-posterior_chain.jld2") chain

    println("Pigeons Sampler Complete. Chain Saved.")
    println("See results/latest for checkpoints.")

    return fpost, pt, chain
end


function dynesty_sample(post, output_dir, snapshot_ID, nlive_points, dlogz; dynamic=false)
    """
    Sample posterior using Dynesty's nested sampler with a specified number
    of live points and dlogz cutoff. Dynesty takes data in unit hypercube form.
    """

    post  # of type Comrade.Posterior
    cpost = ascube(post)

    println("Unit Hypercube Dimensions: $(dimension(cpost)).")

    if dynamic
        smplr = DynamicNestedSampler()
        chain = dysample(post, smplr; nlive_init=1000, dlogz_init=0.01, n_effective=50_000)
    else
        # Create sampler using specified number of live points
        smplr = NestedSampler(;nlive=nlive_points)

        chain = dysample(post, smplr; dlogz=dlogz)
    end

    # Resample the chain to create an equal weighted output
    # Transformed to original space
    equal_weight_chain = Comrade.resample_equal(chain, 10_000)

    # Save chain for later analysis
    @save joinpath(@__DIR__, output_dir, "$snapshot_ID-dynesty-posterior_chain.jld2") equal_weight_chain

    println("Dynesty Sampler Complete. Chain Saved.")

    return cpost, equal_weight_chain
end


function analyze_intmodel(post, chain, output_dir, snapshot_ID)
    """
    Plots for the instrument model gain and phase.
    Less useful for single Sgr A* snapshots.
    """

    # Mean and stdev of gain phases
    mchain = Comrade.rmap(mean, chain);
    schain = Comrade.rmap(std, chain);

    # use Measurements package to plot everything with error bars
    # caltable of variables with errors
    gmeas = instrumentmodel(post, (; instrument = map((x, y) -> Measurements.measurement.(x, y), mchain.instrument, schain.instrument)))
    ctable_am = caltable(abs.(gmeas))
    ctable_ph = caltable(angle.(gmeas))

    # Plot phase curves
    figcaltablephase = plotcaltable(ctable_ph)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-caltable_phase.png"), figcaltablephase)

    figcaltableamp = plotcaltable(ctable_am)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-caltable_amp.png"), figcaltableamp)

    return ctable_am, ctable_ph
end


function image_reconstruction(post, chain, skym, output_dir, snapshot_ID; im_draws=false, fov=200.0, pix=128)
    """
    Reconstruct Images from sampled chain.
    """

    g = imagepixels(μas2rad(fov), μas2rad(fov), pix, pix; executor = ThreadsEx())

    # Image Reconstructions
    samples = skymodel.(Ref(post), chain[begin:5:end])
    #imgs = intensitymap.(samples, Ref(skym.grid))
    imgs = intensitymap.(samples, Ref(g))

    # Array of images
    mimg = mean(imgs)
    simg = std(imgs)
    figim = Figure(; resolution = (700, 700));
    axs = [Axis(figim[i, j], xreversed = true, aspect = 1) for i in 1:2, j in 1:2]
    image!(axs[1, 1], mimg, colormap = :afmhot); axs[1, 1].title = "Mean"
    image!(axs[1, 2], simg ./ (max.(mimg, 1.0e-8)), colorrange = (0.0, 2.0), colormap = :afmhot);axs[1, 2].title = "Std"
    image!(axs[2, 1], imgs[1], colormap = :afmhot);
    image!(axs[2, 2], imgs[end], colormap = :afmhot);
    hidedecorations!.(axs)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-image_reconstruction.png"), figim)

    # Plot Mean and Standard Deviation images
    figmeanimg = imageviz(mimg)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-meanimg.png"), figmeanimg)

    figstdimg = imageviz(simg)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-stdimg.png"), figstdimg)

    if im_draws
       # Image Samples
       mkpath(joinpath(@__DIR__, output_dir, "img_draws/"))

       for (idx, img) in enumerate(imgs[begin:10:end])
          figim = imageviz(img)  # grab the grid from skym
          save(joinpath(@__DIR__, output_dir, "img_draws/$snapshot_ID-img-$idx.png"), figim)
       end
    end

    println("Image reconstruction complete.")

    return mimg, simg
end


function plot_posterior_density(chain, fields, labels, titles, output_dir, snapshot_ID)
    """
    Plot Posterior Density of selected parameters/fields

    e.g. field = rad2μas(chain.sky.r) * 2, label = "Ring Diameter (μas)"
    """

    for (idx, field) in enumerate(fields)
        figpost = Figure(; resolution=(650, 400))
        p = density(figpost[1, 1], field, axis = (xlabel = labels[idx],))

        #ax = Axis(figpost[1, 1]; xlabel = labels[idx])
        #density!(ax, field)
        #figpost[1, 1] = ax
        save(joinpath(@__DIR__, output_dir, "$snapshot_ID-postdens_$(titles[idx]).png"), figpost)
        println("Posterior $(titles[idx]) plotted.")
    end
end


function plot_residuals(post, chain, output_dir)
    """
    Plot normalized residuals
    """

    rd = residuals(post, chain[end])
    fig, ax = plotfields(rd[1], uvdist, :res, axis_kwargs = (; ylabel = "Norm. Res. Vis. Amplitude"))
    plotfields!(fig[2, 1], rd[2], uvdist, :res, axis_kwargs = (; ylabel = "Norm. Res. Vis. Phase"))

    #res_plot = plotfields(res[1], :uvdist, :res)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-residuals.png"), fig)
    println("Residuals plotted.")
end


function plot_corners(chain, fields, title, output_dir, snapshot_ID)
    """
    Plot pair/corner plots of desired parameters

    Title to differentiate if multiple pairplots with different parameter
    subsets are desired.

    e.g. fields = [:a, :incl]

    """

    sky_chain = chain.sky
    println("Sky Chain Parameters: $(propertynames(sky_chain))")
    println("Sky Chain Length: $(length(sky_chain))")
    println("Sky Chain Dimensions: $(ndims(sky_chain))")
    subset = NamedTuple{Tuple(fields)}(Tuple((getproperty(sky_chain, f) for f in fields)))
    pplt = pairplot(subset)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-pairplot_$title.png"), pplt)

end


function plot_coverage(output_dir, data_path)
    """
    Plot u-v coverage

    WARNING: requires Pyehtim loaded
    """

    obseht = ehtim.obsdata.load_uvfits(data_path)
    #obs = Pyehtim.scan_average(obseht)
    coh = extract_table(obs, Coherencies())
    plt = plotfields(coh, U, V, axis_kwargs = (xreversed = true,))
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-uv_coverage.png"), plt)
end

"""
Main
"""

function main(args)
    #model_path = joinpath((@__DIR__), "models.jl")
    #include(model_path)

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

    println("Arguments: $args")
    println("Field of View: $fov μas x $fov μas.")
    println("Pixel Resolution: $pix x $pix.")
    println("Sampler: $sampler.")
    println("Number of Live Points for Dynesty: $nlive_points.")
    println("dlogz for Dynesty: $dlogz.")

    data_path = joinpath((@__DIR__), "snapshot-modeling/scan157_pigeons/scan157-posterior_chain.jld2")
    mpi_run = joinpath((@__DIR__), "results/all/2025-07-22-14-28-36-oSw4GHoe")

    """ 
    Load Data
    """

    if isempty(args)
        snapshot_filepath = joinpath(@__DIR__, "data_comrade",
            "hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan157.jls")
    else
        snapshot_filepath = args[1]
    end

    #if task == "sampling"
    if sampler == "dynesty"
        dvis, output_dir, snapshot_ID = load_data(snapshot_filepath, "nlive-$(nlive_points)_dlogz-$(dlogz)_deltaMD_dynamic_4")
    elseif sampler == "pigeons"
        dvis, output_dir, snapshot_ID = load_data(snapshot_filepath, "pigeons")
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
        g, skym, intmodel, post = setup_sky_model(fov, pix, dvis, skyKrang, priorKrang)
    elseif model == "mG-Ring"
        g, skym, intmodel, post = setup_sky_model(fov, pix, dvis, skyRing, priorRing)
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
    Gather Sampling data/chain for analysis.
    """

    if task == "sampling"
        if sampler == "pigeons"

            println("About to start pigeons sampling...")

            if cluster_config
                fpost, pt, chain = pigeons_cluster(post, output_dir, snapshot_ID,
                                                    mpi=mpi, n_tempering_levels=n_tempering_levels,
                                                    n_threads=n_threads, n_rounds=n_rounds)

                println("Finished pigeons sampling.")
            else
                fpost, pt, chain = pigeons_sample(post, output_dir, snapshot_ID,
                                                    mpi=mpi, n_tempering_levels=n_tempering_levels,
                                                    n_threads=n_threads, n_rounds=n_rounds)
                println("Finished pigeons sampling.")
            end

        elseif sampler == "dynesty"
            cpost, chain = dynesty_sample(post, output_dir, snapshot_ID, nlive_points, dlogz, dynamic=dynamic)

        end
    elseif task == "analysis"
        if sampler == "dynesty"
            println("Data Path: $data_path.")
            #@load data_path equal_weight_chain
    
            data = JLD2.load(data_path)
            chain = data["equal_weight_chain"]

            println("Chain read in.")
        elseif sampler == "pigeons"
            if analysis_source == "chain"
                println("Data Path: $data_path.")
    
                data = JLD2.load(data_path)
                chain = data["chain"]

                println("Chain read in.")
            elseif analysis_source == "pt"
                # Flatten parameter space and move from constrained parameters to
                # (-∞, ∞) support using 'asflat'
                fpost = asflat(post)
                ndim = dimension(fpost)
                println("Flat Space Dimensions: $ndim")

                pt = Pigeons.PT(mpi_run)

                # Save
                @save joinpath(@__DIR__, output_dir, "$snapshot_ID-pt.jld2") pt

                # Transforms back into parameter space with 'NamedTuple' format
                # Posterior chain matches units in prior
                chain = sample_array(fpost, pt)

                # Save transformed chain
                @save joinpath(@__DIR__, output_dir, "$snapshot_ID-posterior_chain.jld2") chain

                println("Pigeons Sampler Complete. Chain Saved.")
                println("See results/latest for checkpoints.")
            end
        end
    end

    """
    Post Analysis
    """

    # Image Reconstruction
    mimg, simg = image_reconstruction(post, chain, skym, output_dir, snapshot_ID; img_draws=true, fov=200.0, pix=128)

    # Posterior Density Plots
    if model == "krang"

        fields = [
            chain.sky.a,
            chain.sky.pa * 180/π,
            #chain.sky.mod,
            chain.sky.incl * 180/π,
            chain.sky.θs * 180/π,
            chain.sky.p1,
            chain.sky.p2,
            chain.sky.rpeak,
            chain.sky.χ * 180/π,
            chain.sky.ι * 180/π,
            chain.sky.βv,
            chain.sky.spec,
            chain.sky.η * 180/π
        ]

        labels = [
            "Spin Parameter a",
            "Position Angle of Projected Spin Axis [Deg]",
            #"Mass-Distance Ratio θg [μas]",
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
            #"mod",
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

    elseif model == "mG-Ring"

        fields = [
            rad2μas(chain.sky.r) * 2,
            rad2μas(chain.sky.σg) * 2 * sqrt(2 * log(2)),
            -rad2deg.(chain.sky.mp.:1) .+ 360.0,
            2 * chain.sky.ma.:2,
            1 .- chain.sky.f
        ]

        labels = [
            "Ring Diameter (μas)",
            "Ring FWHM (μas)",
            "Ring PA (deg) E of N",
            "Brightness asymmetry",
            "Ring flux fraction"
        ]

        titles = ["d", "FWHM", "PA", "BAsym", "RingFlux"]

    end

    plot_posterior_density(chain, fields, labels, titles, output_dir, snapshot_ID)

    if model == "krang"
        pair_fields = (:a, :pa, :incl, :θs, :p1, :p2,
            :rpeak, :χ, :ι, :βv, :spec, :η) # no :mod

    elseif model == "mG-Ring"
        pair_fields = (:r, :w, :ma, :mp, :τ, :ξτ, :f, :σg, :τg,
            :ξg, :xg, :yg)
    end

    plot_corners(chain, pair_fields, "all", output_dir, snapshot_ID)

    #plot_residuals(post, chain, output_dir)

    #exit()

end

end

if abspath(PROGRAM_FILE) == @__FILE__
    SnapshotModeling.main(ARGS)
end