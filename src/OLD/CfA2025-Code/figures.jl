using CairoMakie
using PairPlots
using JLD2
using CSV
using Glob
using DataFrames
using Statistics
using Distributions
using Dates
using LaTeXStrings
using KernelDensity


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

function plot_corners(chain, fields, title, output_dir, snapshot_ID, labels)
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

    sky_chain = chain.sky
    println("Sky Chain Parameters: $(propertynames(sky_chain))")
    println("Sky Chain Length: $(length(sky_chain))")
    println("Sky Chain Dimensions: $(ndims(sky_chain))")
    subset = NamedTuple{Tuple(fields)}(Tuple((getproperty(sky_chain, f) for f in fields)))
    pplt = pairplot(subset; var_names=labels)
    save(joinpath(@__DIR__, output_dir, "$snapshot_ID-pairplot_$title.png"), pplt)

    set_theme!()

    #set_theme!(Theme(
    #    Axis=(xlabelsize=20, ylabelsize=20, titlesize=22),
    #))

end

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


function ridge_plot(chainpaths, scans, times)
    #f = Figure()
    #Axis(f[1, 1])

    #fields = [
    #    chain.sky.a,
    #    chain.sky.pa * 180/π,
        #chain.sky.mod,
    #    chain.sky.incl * 180/π,
    #    chain.sky.θs * 180/π,
    #    chain.sky.p1,
    #    chain.sky.p2,
    #    chain.sky.rpeak,
    #    chain.sky.χ * 180/π,
    #    chain.sky.ι * 180/π,
    #    chain.sky.βv,
    #    chain.sky.spec,
    #    chain.sky.η * 180/π
    #]

    #for (idx, field) in enumerate(fields)
    #f = Figure(; resolution=(650, 400))
    f = Figure()
    Axis(f[1, 1], yticks = ((1:11) ./ 4, scans[begin:15:end]))

    for (i, chainpath) in enumerate(chainpaths[begin:15:end])
        data = JLD2.load(chainpath)
        chain = data["chain"]

        d = density!(chain.sky.a, offset = -i / 4,
            color = (:slategray, 0.4), bandwidth = 0.1)
            #color = :x, colormap = :thermal, colorrange = (-5, 5),
            #strokewidth = 1, strokecolor = :black)
        # this helps with layering in GLMakie
        #translate!(d, 0, 0, -0.1i)

        #density!(chain.sky.a, offset = -i/4, color = (:slategray, 0.4),
        #    bandwidth = 0.1)
    end
    #p = density(figpost[1, 1], field, axis = (xlabel = labels[idx],))
    #ax = Axis(figpost[1, 1]; xlabel = labels[idx])
    #density!(ax, field)
    #figpost[1, 1] = ax
    save(joinpath(@__DIR__, "ridge-plots/", "ridge-plot_a.png"), f)
    #println("Posterior $(titles[idx]) plotted.")
    #end
end



function ridge_plot2(chainpaths, scans, times)
    f = Figure(resolution = (800, 600))
    
    # subsample for readability
    idxs = 1:15:length(chainpaths)
    
    # Axis with y labels at the offsets
    offsets = -collect(1:length(idxs)) ./ 4
    Axis(f[1, 1],
         yticks = (offsets, string.(scans[idxs])),
         xlabel = "a",
         ylabel = "Scan")

    for (i, chainpath) in enumerate(chainpaths[idxs])
        data = JLD2.load(chainpath)
        chain = data["chain"]

        density!(chain.sky.a,
                 offset = offsets[i],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1)
    end

    save(joinpath(@__DIR__, "ridge-plots", "ridge-plot_a.png"), f)
    return f
end


function ridge_plot_csv(csvpaths, scans, times; field=:a, output="ridge_a.png")
    f = Figure(resolution = (800, 1200))
    # subsample for readability
    idxs = 1:10:length(csvpaths)
    n = length(idxs)
    offsets = -collect(1:n) ./ 4

    ax = Axis(f[1, 1];
              yticks = (offsets, string.(round.(times[idxs], digits=2))),
              xlabel = string(field),
              ylabel = "Time [MJD]")

    for (j, i) in enumerate(idxs)
        df = CSV.read(csvpaths[i], DataFrame)

        values = df[!, field]

        density!(ax, values;
                 offset = offsets[j],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1)
    end

    save(joinpath(@__DIR__, "ridge-plots", output), f)
    return f
end




"""
    ridge_plot_with_gaussians(csv_files; gaussians=[], subsample=10, outpath="ridgeplot.png")

Creates a stacked ridge plot of sky-chain samples from CSV files, grouped by day (MJD → UTC).
A Gaussian or series of Gaussians can be plotted on top, aligned in time.

# Arguments
- `csv_files::Vector{String}` : Paths to CSV skychain files (must include `:a` and `:time` in MJD).
- `gaussians::Vector{Tuple{Float64,Float64,Float64}}` : Optional list of (μ, σ, amp) in MJD.
- `subsample::Int` : Subsampling factor for scans to avoid overcrowding.
- `outpath::String` : Path to save the PNG figure.
"""
function ridge_plot_with_gaussians(csv_files, scans, times; gaussians=[], subsample=10, outpath="ridgeplot.png")
    # --- helper: MJD → DateTime UTC
    function mjd_to_datetime(mjd::Float64)
        dt0 = DateTime(1858,11,17) # MJD reference
        return dt0 + Millisecond(round(Int, mjd*86400000)) # mjd*24*3600*1000
    end

    # --- load CSVs
    dfs = DataFrame[]
    for file in csv_files
        df = CSV.read(file, DataFrame)
        push!(dfs, df)
    end
    df_all = vcat(dfs...)

    # require :time (MJD) and :a (parameter for density)
    #if !(:time in names(df_all)) || !(:a in names(df_all))
    #    error("CSV files must contain columns :time (MJD) and :a (sky-chain parameter)")
    #end

    # --- convert time
    df_all.datetime = mjd_to_datetime.(df_all.time)
    df_all.day = Date.(df_all.datetime)

    # --- split by day
    by_day = groupby(df_all, :day)

    # --- set up figure
    n_days = length(by_day)
    f = Figure(resolution=(1200, 350*n_days + 300))
    gl = GridLayout(f)

    # --- top panel for Gaussians
    ax_top = Axis(gl[1,1], xticksvisible=false, yticksvisible=false,
                  xgridvisible=false, ygridvisible=false,
                  xlabel="Time (UTC)", ylabel="", title="Gaussians")

    # plot Gaussians
    if !isempty(gaussians)
        xs = range(minimum(df_all.time), maximum(df_all.time), length=800)
        for (μ, σ, amp) in gaussians
            ys = amp .* pdf.(Normal(μ, σ), xs)
            lines!(ax_top, mjd_to_datetime.(xs), ys, color=:red, linewidth=2)
        end
    end

    # --- ridge plots per day
    axes = Axis[]
    for (i, g) in enumerate(by_day)
        # extract unique scans for this day
        scans = unique(g.scan)

        # make labels in UTC time
        scan_times = Dict{Any,String}()
        for s in scans
            t = first(g[g.scan .== s, :datetime])
            scan_times[s] = Dates.format(t, "HH:MM")
        end

        # y-ticks = evenly spaced offsets
        yticks = ((1:length(scans[1:subsample:end])) ./ 4,
                  [scan_times[s] for s in scans[1:subsample:end]])

        ax = Axis(gl[i+1,1], ylabel="UTC", yticks=yticks,
                  title="Day $(first(g.day))")

        for (j, s) in enumerate(scans[1:subsample:end])
            gsub = filter(:scan => ==(s), g)
            density!(ax, gsub.a; offset=-j/4,
                     color=(:slategray, 0.4), bandwidth=0.1)
        end

        hidexdecorations!(ax; grid=false) # keep only top axis for time
        push!(axes, ax)
    end

    # align all x-axes
    linkxaxes!(ax_top, axes)

    save(outpath, f)
    println("Saved ridge plot with Gaussians → $outpath")
    return f
end


function ridge_plot_with_truncation_and_gaussians(csv_files; 
        field=:a, xlim=(-1.0, 1.0), gaussians=[], subsample=10, outpath="ridgeplot.png")

    # Helper: MJD → DateTime UTC
    mjd_to_datetime(mjd::Float64) = DateTime(1858,11,17) + Millisecond(round(Int, mjd*86400000))

    # Load all CSVs into one DataFrame
    dfs = [CSV.read(f, DataFrame) for f in csv_files]
    df_all = vcat(dfs...)
    
    # Convert time
    df_all[:, :datetime] = mjd_to_datetime.(df_all.time)
    df_all[:, :day] = Date.(df_all.datetime)
    
    # Group by day
    by_day = groupby(df_all, :day)
    
    n_days = length(by_day)
    f = Figure(resolution=(1200, 350*n_days + 300))
    gl = GridLayout(f)
    
    # --- Top panel: Gaussians
    ax_top = Axis(gl[1,1], xticksvisible=false, yticksvisible=false, xgridvisible=false, ygridvisible=false,
                  xlabel="Time (UTC)", ylabel="", title="Gaussians")
    
    if !isempty(gaussians)
        xs = range(xlim[1], xlim[2], length=800)
        for (μ, σ, amp) in gaussians
            ys = amp .* pdf.(Normal(μ, σ), xs)
            lines!(ax_top, xs, ys, color=:red, linewidth=2)
        end
    end

    axes = Axis[]
    for (i, g) in enumerate(by_day)
        # Unique scans for the day
        scans_day = unique(g.scan)
        
        # Make UTC labels
        scan_times = Dict(s => Dates.format(first(g[g.scan .== s, :datetime]), "HH:MM") for s in scans_day)
        
        # Subsample scans for readability
        scans_sub = scans_day[1:subsample:end]
        offsets = -collect(1:length(scans_sub)) ./ 4
        ylabels = [scan_times[s] for s in scans_sub]
        
        ax = Axis(gl[i+1, 1], ylabel="UTC", yticks=(offsets, ylabels), title="Day $(first(g.day))", xlabel=string(field))
        
        for (j, s) in enumerate(scans_sub)
            gsub = filter(:scan => ==(s), g)
            # truncate values to xlim
            vals = clamp.(gsub[!, field], xlim[1], xlim[2])
            density!(ax, vals; offset=offsets[j], color=(:slategray, 0.4), bandwidth=0.1)
        end
        push!(axes, ax)
    end
    
    # Align x axes
    linkxaxes!(ax_top, axes)
    xlims!(ax_top, xlim...)
    
    save(outpath, f)
    println("Saved ridge plot → $outpath")
    return f
end


function ridge_plot_with_scans_and_times(
        csv_files::Vector{String}, 
        scans::Vector, 
        times::Vector{Float64}; 
        field=:a, 
        xlim=(-1.0, 1.0), 
        gaussians=[], 
        subsample=10, 
        outpath="ridgeplot.png",
        xlabel::Union{String, LaTeXString}=string(field)
    )

    # Helper: MJD → DateTime UTC
    mjd_to_datetime(mjd::Float64) = DateTime(1858,11,17) + Millisecond(round(Int, mjd*86400000))

    n_files = length(csv_files)
    if length(scans) != n_files || length(times) != n_files
        error("Length of scans and times must match number of CSV files")
    end

    # Convert times to UTC DateTime
    datetimes = mjd_to_datetime.(times)
    days = Date.(datetimes)

    # Determine unique days
    unique_days = unique(days)
    n_days = length(unique_days)

    f = Figure(resolution=(1200, 350*n_days + 300))

    # --- Top panel: Gaussians
    ax_top = Axis(f[1,1], xticksvisible=false, yticksvisible=false, xgridvisible=false, ygridvisible=false,
                  xlabel=xlabel, ylabel="", title="Gaussians")

    if !isempty(gaussians)
        xs = range(xlim[1], xlim[2], length=800)
        for (μ, σ, amp) in gaussians
            ys = amp .* pdf.(Normal(μ, σ), xs)
            lines!(ax_top, xs, ys, color=:red, linewidth=2)
        end
    end

    axes = Axis[]
    for (i, day) in enumerate(unique_days)
        # Get indices for this day
        day_idx = findall(days .== day)
        scans_day = scans[day_idx]
        csvs_day = csv_files[day_idx]
        times_day = datetimes[day_idx]

        # Subsample
        subsample_idx = 1:subsample:length(csvs_day)
        scans_sub = scans_day[subsample_idx]
        csvs_sub = csvs_day[subsample_idx]

        offsets = -collect(1:length(scans_sub)) ./ 4
        ylabels = Dates.format.(times_day[subsample_idx], "HH:MM")

        ax = Axis(f[i+1, 1], ylabel="UTC", yticks=(offsets, ylabels), title="Day $day", xlabel=xlabel)

        for (j, csvfile) in enumerate(csvs_sub)
            df = CSV.read(csvfile, DataFrame)
            vals = clamp.(df[!, field], xlim[1], xlim[2])
            density!(ax, vals; offset=offsets[j], color=(:slategray, 0.4), bandwidth=0.1)
        end

        push!(axes, ax)
    end

    # Align x axes
    #linkxaxes!(ax_top, axes)
    linkxaxes!(ax_top, axes...)
    xlims!(ax_top, xlim...)

    save(outpath, f)
    println("Saved ridge plot → $outpath")
    return f
end


function ridge_plot_expanded(
    csvpaths, scans, times;
    field = :a,
    output = "ridge_a.png",
    posterior_color = (:slategray, 0.4),     # single color or array
    gaussians = [],                          # list of NamedTuples: (μ, σ)
    xlims = nothing,                         # e.g. (xmin, xmax)
    xlabel = nothing,                        # LaTeXString or String
    ylabel = nothing,                        # LaTeXString or String
    title  = nothing,                        # LaTeXString or String
    yaxis = :mjd                             # :mjd | :scan | :utc
)

    # Determine y-values based on yaxis mode
    if yaxis == :scan
        yvals = scans
        ylabels = string.(scans)
    elseif yaxis == :mjd
        yvals = times
        ylabels = string.(round.(times, digits = 2))
    elseif yaxis == :utc
        # assume `times` are in MJD; convert to DateTime
        base = DateTime(1858,11,17,0,0,0)
        datetimes = [base + Dates.Day(t) for t in times]
        days = Dates.value.(Dates.date.(datetimes))
        unique_days = unique(days)
        # Group indices by day
        day_groups = Dict(day => findall(d -> d == day, days) for day in unique_days)
    else
        error("Invalid yaxis setting: choose :mjd, :scan, or :utc")
    end

    # Set up figure(s)
    if yaxis == :utc
        n_panels = length(keys(day_groups))
        f = Figure(resolution = (800, 400 * n_panels))
        panel_index = 1
    else
        f = Figure(resolution = (800, 1200))
    end

    # X-limits
    if xlims !== nothing
        xmin, xmax = xlims
    else
        xmin, xmax = nothing, nothing
    end

    # Function to plot ridges for a group of indices
    function plot_ridges(ax, indices, label_vals)
        idxs = 1:10:length(indices)  # subsample
        n = length(idxs)
        offsets = -collect(1:n) ./ 4

        # yticks
        yticks = (offsets, string.(label_vals[idxs]))
        ax.yticks = yticks

        # posterior color handling
        use_colors = isa(posterior_color, AbstractVector) ? posterior_color : fill(posterior_color, n)

        for (j, local_i) in enumerate(idxs)
            global_i = indices[local_i]
            df = CSV.read(csvpaths[global_i], DataFrame)
            values = df[!, field]
            density!(ax, values;
                offset = offsets[j],
                color = use_colors[j],
                bandwidth = 0.1)
        end
    end

    if yaxis == :utc
        for (d, indices) in sort(collect(day_groups); by = x -> x[1])
            datetimes = [DateTime(1858,11,17) + Dates.Day(times[i]) for i in indices]
            day_label = string(Dates.date(datetimes[1]))
            ax = Axis(f[panel_index, 1]; ylabel = day_label)
            plot_ridges(ax, indices, datetimes)

            if xlabel !== nothing; ax.xlabel = xlabel; end
            if xlims !== nothing; xlims!(ax, xmin, xmax); end

            panel_index += 1
        end
    else
        indices = 1:length(times)
        ax = Axis(f[1, 1])
        plot_ridges(ax, indices, yvals)

        if xlabel !== nothing; ax.xlabel = xlabel; else; ax.xlabel = string(field); end
        if ylabel !== nothing
            ax.ylabel = ylabel
        else
            ax.ylabel = yaxis == :mjd ? "Time [MJD]" :
                        yaxis == :scan ? "Scan Number" :
                        ""
        end
        if title !== nothing; ax.title = title; end
        if xlims !== nothing; xlims!(ax, xmin, xmax); end
    end

    # Gaussian overlays at top
    if !isempty(gaussians) && yaxis != :utc
        ax = content(f)[1]  # reference first/only axis
        for g in gaussians
            μ = g[:μ]; σ = g[:σ]
            if xlims === nothing
                xs = range(μ - 4σ, μ + 4σ, length = 400)
            else
                xs = range(xmin, xmax, length = 400)
            end
            ys = @. exp(-0.5 * ((xs - μ) / σ)^2)  # unit max
            lines!(ax, xs, ys .+ 0.2)  # shift above ridges
        end
    end

    save(joinpath(@__DIR__, "ridge-plots", output), f)
    return f
end

function ridge_plot_expanded_2(
    csvpaths, scans, times;
    field = :a,
    output = "ridge_a.png",
    posterior_color = (:slategray, 0.4),     # single color or array
    gaussians = [],                          # list of NamedTuples: (μ, σ)
    xlims = nothing,                         # e.g. (xmin, xmax)
    xlabel = nothing,                        # LaTeXString or String
    ylabel = nothing,                        # LaTeXString or String
    title  = nothing,                        # LaTeXString or String
    yaxis = :mjd                             # :mjd | :scan | :utc
)

    mjd_to_datetime(mjd::Float64) = DateTime(1858,11,17) + Millisecond(round(Int, mjd*86400000))

    # Determine y-values based on yaxis mode
    if yaxis == :scan
        yvals = scans
        ylabels = string.(scans)
    elseif yaxis == :mjd
        yvals = times
        ylabels = string.(round.(times, digits = 2))
    elseif yaxis == :utc
        base = DateTime(1858,11,17,0,0,0)
        #datetimes = [base + Dates.Day(trunc(t)) for t in times]
        datetimes = mjd_to_datetime.(times)
        days = Date.(datetimes)
        #days = Dates.value.(Dates.date.(datetimes))
        unique_days = unique(days)
        day_groups = Dict(day => findall(d -> d == day, days) for day in unique_days)
    else
        error("Invalid yaxis setting: choose :mjd, :scan, or :utc")
    end

    # Set up figure(s)
    if yaxis == :utc
        n_panels = length(keys(day_groups))
        f = Figure(resolution = (800, 400 * n_panels))
        panel_index = 1
    else
        f = Figure(resolution = (800, 1200))
    end

    # X-limits
    if xlims !== nothing
        xmin, xmax = xlims
    else
        xmin, xmax = nothing, nothing
    end

    # Local helper to plot ridges for a set of indices
    function plot_ridges(ax, indices, label_vals)
        idxs = 1:10:length(indices)  # subsample
        n = length(idxs)
        offsets = -collect(1:n) ./ 4

        yticks = (offsets, string.(label_vals[idxs]))
        ax.yticks = yticks

        use_colors = isa(posterior_color, AbstractVector) ? posterior_color : fill(posterior_color, n)

        for (j, local_i) in enumerate(idxs)
            global_i = indices[local_i]
            df = CSV.read(csvpaths[global_i], DataFrame)
            values = df[!, field]
            density!(ax, values;
                offset = offsets[j],
                color = use_colors[j],
                bandwidth = 0.1)
        end
    end

    # ===== Plotting logic =====
    if yaxis == :utc
        for (d, indices) in sort(collect(day_groups); by = x -> x[1])
            ax = Axis(f[panel_index, 1]; ylabel = string(Date(DateTime(1858,11,17) + Dates.Day(trunc.(times[indices[1]])))))
            datetimes = [DateTime(1858,11,17) + Dates.Day(trunc.(times[i])) for i in indices]
            plot_ridges(ax, indices, datetimes)

            if xlabel !== nothing; ax.xlabel = xlabel; end
            if xlims !== nothing; xlims!(ax, xmin, xmax); end

            panel_index += 1
        end

    else
        indices = 1:length(times)
        ax = Axis(f[1, 1])
        plot_ridges(ax, indices, yvals)

        if xlabel !== nothing
            ax.xlabel = xlabel
        else
            ax.xlabel = string(field)
        end

        if ylabel !== nothing
            ax.ylabel = ylabel
        else
            ax.ylabel = yaxis == :mjd ? "Time [MJD]" :
                        yaxis == :scan ? "Scan Number" :
                        ""
        end

        if title !== nothing
            ax.title = title
        end

        if xlims !== nothing
            xlims!(ax, xmin, xmax)
        end

        # ✅ Gaussian overlays placed here using the same `ax`
        if !isempty(gaussians)
            for g in gaussians
                μ = g[:μ]; σ = g[:σ]
                if xlims === nothing
                    xs = range(μ - 4σ, μ + 4σ, length = 400)
                else
                    xs = range(xmin, xmax, length = 400)
                end
                ys = @. exp(-0.5 * ((xs - μ) / σ)^2)  # unit peak
                lines!(ax, xs, ys .+ 0.2)
            end
        end
    end

    save(joinpath(@__DIR__, "ridge-plots", output), f)
    return f
end




function ridgeplot(
    csvpaths, scans, times;
    field = :a,
    output = "ridge_a.png",
    posterior_color = (:slategray, 0.4),     # single color or array
    gaussians = [],                          # list of NamedTuples: (μ, σ)
    xlims = nothing,                         # e.g. (xmin, xmax)
    xlabel = nothing,                        # LaTeXString or String
    ylabel = nothing,                        # LaTeXString or String
    title  = nothing,                        # LaTeXString or String
    yaxis = :mjd                             # :mjd | :scan | :utc
)

    mjd_to_datetime(mjd::Float64) = DateTime(1858,11,17) + Millisecond(round(Int, mjd*86400000))

    # Determine y-values based on yaxis mode
    if yaxis == :scan
        yvals = scans
        ylabels = string.(scans)
    elseif yaxis == :mjd
        yvals = times
        ylabels = string.(round.(times, digits = 2))
    elseif yaxis == :utc
        base = DateTime(1858,11,17,0,0,0)
        #datetimes = [base + Dates.Day(trunc(t)) for t in times]
        datetimes = mjd_to_datetime.(times)
        days = Date.(datetimes)
        #days = Dates.value.(Dates.date.(datetimes))
        unique_days = unique(days)
        day_groups = Dict(day => findall(d -> d == day, days) for day in unique_days)
    else
        error("Invalid yaxis setting: choose :mjd, :scan, or :utc")
    end

    # Set up figure(s)
    if yaxis == :utc
        n_panels = length(keys(day_groups))
        f = Figure(resolution = (800, 400 * n_panels))
        panel_index = 1
    else
        f = Figure(resolution = (800, 1200))
    end

    # X-limits
    if xlims !== nothing
        xmin, xmax = xlims
    else
        xmin, xmax = nothing, nothing
    end

    # Local helper to plot ridges for a set of indices
    function plot_ridges(ax, indices, label_vals)
        idxs = 1:10:length(indices)  # subsample
        n = length(idxs)
        offsets = -collect(1:n) ./ 4

        yticks = (offsets, string.(label_vals[idxs]))
        ax.yticks = yticks

        use_colors = isa(posterior_color, AbstractVector) ? posterior_color : fill(posterior_color, n)

        for (j, local_i) in enumerate(idxs)
            global_i = indices[local_i]
            df = CSV.read(csvpaths[global_i], DataFrame)
            values = df[!, field]
            density!(ax, values;
                offset = offsets[j],
                color = use_colors[j],
                bandwidth = 0.1)
        end
    end

    # ===== Plotting logic =====
    if yaxis == :utc
        for (d, indices) in sort(collect(day_groups); by = x -> x[1])
            ax = Axis(f[panel_index, 1]; ylabel = string(Date(DateTime(1858,11,17) + Dates.Day(trunc.(times[indices[1]])))))
            datetimes = [DateTime(1858,11,17) + Dates.Day(trunc.(times[i])) for i in indices]
            plot_ridges(ax, indices, datetimes)

            if xlabel !== nothing; ax.xlabel = xlabel; end
            if xlims !== nothing; xlims!(ax, xmin, xmax); end

            panel_index += 1
        end

    else
        indices = 1:length(times)
        ax = Axis(f[1, 1])
        plot_ridges(ax, indices, yvals)

        if xlabel !== nothing
            ax.xlabel = xlabel
        else
            ax.xlabel = string(field)
        end

        if ylabel !== nothing
            ax.ylabel = ylabel
        else
            ax.ylabel = yaxis == :mjd ? "Time [MJD]" :
                        yaxis == :scan ? "Scan Number" :
                        ""
        end

        if title !== nothing
            ax.title = title
        end

        if xlims !== nothing
            xlims!(ax, xmin, xmax)
        end

        # ✅ Gaussian overlays placed here using the same `ax`
        if !isempty(gaussians)
            for g in gaussians
                μ = g[:μ]; σ = g[:σ]
                if xlims === nothing
                    xs = range(μ - 4σ, μ + 4σ, length = 400)
                else
                    xs = range(xmin, xmax, length = 400)
                end
                ys = @. exp(-0.5 * ((xs - μ) / σ)^2)  # unit peak
                lines!(ax, xs, ys .+ 0.2)
            end
        end
    end

    save(joinpath(@__DIR__, "ridge-plots", output), f)
    return f
end



function ridge_plot_csv_3(csvpaths, scans, times; 
                        field=:a, 
                        output="ridge_a.png", 
                        utc=false)

    f = Figure(resolution = (800, 1200))

    # subsample for readability
    idxs = 1:10:length(csvpaths)
    n = length(idxs)
    offsets = -collect(1:n) ./ 4

    # convert times if UTC requested
    ticklabels = String[]
    if utc
        mjd0 = DateTime(1858, 11, 17)  # MJD epoch
        for t in times[idxs]
            dt = mjd0 + Dates.Day(floor(Int, t)) + Dates.Millisecond(round(Int, (t % 1) * 86400000))
            push!(ticklabels, string(Dates.hour(dt), ":00 UTC\n", Dates.format(dt, "yyyy-mm-dd")))
        end
        ylabel = "Time [UTC]"
    else
        ticklabels = string.(round.(times[idxs], digits=2))
        ylabel = "Time [MJD]"
    end

    ax = Axis(f[1, 1];
              yticks = (offsets, ticklabels),
              xlabel = string(field),
              ylabel = ylabel)

    for (j, i) in enumerate(idxs)
        df = CSV.read(csvpaths[i], DataFrame)

        values = df[!, field]  # get column as vector

        density!(ax, values;
                 offset = offsets[j],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1)
    end

    save(joinpath(@__DIR__, "ridge-plots", output), f)
    return f
end




function ridge_plot_csv_4(csvpaths, scans, times;
                        lims,
                        xlabel,
                        field=:a,
                        output="ridge_a.png",
                        utc=false,
                        angular=false)

    #theme = Theme(fontsize = 20)
    #set_theme!(theme)
    set_theme!(Makie.theme_latexfonts())
    update_theme!(fontsize=25)

    f = Figure(resolution = (800, 1200))

    idxs = 1:10:length(csvpaths)
    #idxs = 1:2:length(csvpaths)
    n = length(idxs)
    offsets = -collect(1:n) ./ 5

    ticklabels = String[]
    datelabel = ""
    dts = DateTime[]
    if utc
        mjd0 = DateTime(1858, 11, 17)
        for t in times[idxs]
            dt = mjd0 + Dates.Day(floor(Int, t)) + Dates.Millisecond(round(Int, (t % 1) * 86_400_000))
            push!(dts, dt)
            push!(ticklabels, Dates.format(dt, "HH:MM"))
        end
        datelabel = Dates.format(dts[1], "yyyy-mm-dd")
        ylabel = "Time [UTC]"
    else
        ticklabels = string.(round.(times[idxs], digits=2))
        ylabel = "Time [MJD]"
    end

    ax = Axis(f[1, 1];
              yticks = (offsets, ticklabels),
              xlabel = xlabel,
              ylabel = ylabel,
              limits= (lims, nothing),
              xlabelsize=28)

    for (j, i) in enumerate(idxs)
        df = CSV.read(csvpaths[i], DataFrame)
        if angular
            values = (180 / π) .* df[!, field]

            kd = kde(values)
            y_corr = kd.density ./ (180 / π)
            poly!(ax, kd.x, y_corr .+ offsets[j], color = (:slategray, 0.5))
            lines!(ax, kd.x, y_corr .+ offsets[j], color = :gray, linewidth = 2)

            #density!(ax, values;
            #     offset = offsets[j],
            #     color = (:slategray, 0.4),
            #     bandwidth = 0.1,
            #     strokecolor=:black,
            #     strokewidth=1)
        else
            values = df[!, field]
            density!(ax, values;
                 offset = offsets[j],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1,
                 strokecolor=:black,
                 strokewidth=1)
        end
                 #boundary=lims)

        #xlims!(ax, lims[1], lims[2])
    end

    # Add date label inside plot using annotation
    #if utc
    #    annotation!(ax, datelabel, 0.02, 0.98,
    #                align = (:left, :top),
    #                fontsize = 16)
    #end
    if utc
        text!(ax, datelabel,
          position = (0.01, 0.99), # x and y relative to the scene
          space = :relative,      # Coordinates are relative to the scene
          align = (:left, :top),  # Align the text to the top-left corner
          fontsize = 28)
    end

    mkpath(joinpath(@__DIR__, "ridge-plots-grmhdtest3599-16-12-2025"))
    save(joinpath(@__DIR__, "ridge-plots-grmhdtest3599-16-12-2025", output), f)

    set_theme!()

    return f
end


function ridgeplot_gaussians(csvpaths, scans, times;
                        lims,
                        xlabel,
                        field=:a,
                        output="ridge_a.png",
                        utc=false,
                        angular=false,
                        gaussians = [])
                        #gaussians::Vector{Tuple{Float64, Float64}} = [])

    #theme = Theme(fontsize = 20)
    #set_theme!(theme)
    set_theme!(Makie.theme_latexfonts())
    update_theme!(fontsize=25)

    idxs = 1:7:length(csvpaths)
    n = length(idxs)
    offsets = -collect(1:n) ./ 5

    ticklabels = String[]
    datelabel = ""
    dts = DateTime[]
    if utc
        mjd0 = DateTime(1858, 11, 17)
        for t in times[idxs]
            dt = mjd0 + Dates.Day(floor(Int, t)) + Dates.Millisecond(round(Int, (t % 1) * 86_400_000))
            push!(dts, dt)
            push!(ticklabels, Dates.format(dt, "HH:MM"))
        end
        datelabel = Dates.format(dts[1], "yyyy-mm-dd")
        ylabel = "Time [UTC]"
    else
        ticklabels = string.(round.(times[idxs], digits=2))
        ylabel = "Time [MJD]"
    end

    f = Figure(resolution = (800, 1200))

    ga = f[1, 1] = GridLayout()
    axgaussian = Axis(ga[1, 1])
    axridge = Axis(ga[2, 1];
                yticks = (offsets, ticklabels),
                xlabel = xlabel,
                ylabel = ylabel,
                limits= (lims, nothing),
                xlabelsize=28
            )

    linkxaxes!(axgaussian, axridge)

    for (j, i) in enumerate(idxs)
        df = CSV.read(csvpaths[i], DataFrame)
        if angular
            values = (180 / π) .* df[!, field]
        else
            values = df[!, field]
        end
        density!(axridge, values;
                 offset = offsets[j],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1,
                 strokecolor=:black,
                 strokewidth=1)
                 #boundary=lims)

        #xlims!(ax, lims[1], lims[2])
    end

    # Add date label inside plot using annotation
    #if utc
    #    annotation!(ax, datelabel, 0.02, 0.98,
    #                align = (:left, :top),
    #                fontsize = 16)
    #end
    if utc
        text!(axridge, datelabel,
          position = (0.01, 0.99), # x and y relative to the scene
          space = :relative,      # Coordinates are relative to the scene
          align = (:left, :top),  # Align the text to the top-left corner
          fontsize = 28)
    end

    if !isempty(gaussians)
        x = range(lims[1], lims[2], length=400)
        # Estimate top offset from last ridge line
        top_offset = offsets[1] + 0.2  # small upward shift above the top ridge
        for (μ, σ) in gaussians
            y = pdf.(Normal(μ, σ), x)
            yscaled = y ./ maximum(y) .* 0.4  # normalized to look nice
            lines!(axgaussian, x, yscaled .+ top_offset, color=:gray, linewidth=2)
        end
    end

    ylims!(axgaussian, low = 0)

    rowsize!(ga, 2, Relative(9/10))
    hidespines!(axgaussian)
    hidedecorations!(axgaussian)
    #hidedecorations!(axgaussian, ticks=false, label=false, ticklabels=false, grid=false)

    save(joinpath(@__DIR__, "ridge-plots-2", output), f)

    set_theme!()

    return f
end


function ridge_plot_guassians(csvpaths, scans, times;
                        lims,
                        xlabel,
                        field=:a,
                        output="ridge_a.png",
                        utc=false,
                        angular=false,
                        gaussians::Vector{Tuple{Float64, Float64}} = [])

    #theme = Theme(fontsize = 20)
    #set_theme!(theme)
    #update_theme!(font = "CMU Serif")

    set_theme!(Makie.theme_latexfonts())
    update_theme!(fontsize=20)

    f = Figure(resolution = (800, 1200))

    idxs = 1:10:length(csvpaths)
    n = length(idxs)
    offsets = -collect(1:n) ./ 5

    ticklabels = String[]
    datelabel = ""
    dts = DateTime[]
    if utc
        mjd0 = DateTime(1858, 11, 17)
        for t in times[idxs]
            dt = mjd0 + Dates.Day(floor(Int, t)) + Dates.Millisecond(round(Int, (t % 1) * 86_400_000))
            push!(dts, dt)
            push!(ticklabels, Dates.format(dt, "HH:MM"))
        end
        datelabel = Dates.format(dts[1], "yyyy-mm-dd")
        ylabel = "Time [UTC]"
    else
        ticklabels = string.(round.(times[idxs], digits=2))
        ylabel = "Time [MJD]"
    end

    # --- Main ridge plot axis ---
    ax = Axis(f[1, 1];
              yticks = (offsets, ticklabels),
              xlabel = xlabel,
              ylabel = ylabel,
              limits = (lims, nothing),
              xlabelsize = 28)

    for (j, i) in enumerate(idxs)
        df = CSV.read(csvpaths[i], DataFrame)
        values = angular ? (180 / π) .* df[!, field] : df[!, field]

        density!(ax, values;
                 offset = offsets[j],
                 color = (:slategray, 0.4),
                 bandwidth = 0.1,
                 strokecolor = :black,
                 strokewidth = 1)
    end

    if utc
        text!(ax, datelabel,
              position = (0.01, 0.99),
              space = :relative,
              align = (:left, :top),
              fontsize = 28)
    end

    # --- Overlay the Gaussian curves directly on same axis ---
    if !isempty(gaussians)
        x = range(lims[1], lims[2], length=400)
        # Estimate top offset from last ridge line
        top_offset = offsets[1] + 0.2  # small upward shift above the top ridge
        for (μ, σ) in gaussians
            y = pdf.(Normal(μ, σ), x)
            yscaled = y ./ maximum(y) .* 0.4  # normalized to look nice
            lines!(ax, x, yscaled .+ top_offset, color=:black, linewidth=2)
        end
    end

    # --- Remove any visible frame/axes lines entirely ---
    hidespines!(ax)
    hidedecorations!(ax, ticks=false, label=false, ticklabels=false, grid=false)

    save(joinpath(@__DIR__, "ridge-plots-2", output), f)
    set_theme!()
    return f
end










#update_theme!(font = "CMU Serif")

set_theme!(Makie.theme_latexfonts())
update_theme!(fontsize=25)

pair_fields = (:a, :pa, :incl, :θs, :p1, :p2,
            :rpeak, :χ, :ι, :βv, :spec, :η)

#fields = [
#    chain.sky.a,
#    chain.sky.pa * 180/π,
    #chain.sky.mod,
#    chain.sky.incl * 180/π,
#    chain.sky.θs * 180/π,
#    chain.sky.p1,
#    chain.sky.p2,
#    chain.sky.rpeak,
#    chain.sky.χ * 180/π,
#    chain.sky.ι * 180/π,
#    chain.sky.βv,
#    chain.sky.spec,
#    chain.sky.η * 180/π
#]

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


#chainpaths = readdir(joinpath((@__DIR__), "../CfA-Proj/stacking/chains"); join=true)
chainpaths = readdir(joinpath((@__DIR__), "../CfA-Proj/stacking/chains-grmhdtest3599"); join=true)

#summary_files = filter(f -> occursin(r"^scan.*-dfsum\.csv$", f), readdir(joinpath((@__DIR__), "../CfA-Proj/stacking/csv-chains-SgrA/"); join=true))

summary_files = glob("scan*-dfsum.csv", (joinpath((@__DIR__), "../CfA-Proj/stacking/csv-chains-grmhdtest3599/")))
#summary_files = glob("scan*-dfsum.csv", (joinpath((@__DIR__), "../CfA-Proj/stacking/csv-chains-SgrA/")))


#println(summary_files)

#df = CSV.read(summary_files[0], DataFrame)
#print(df)

dfsums = [CSV.read(f, DataFrame) for f in summary_files]
#println(first(dfsums))
dfsums_all = vcat(dfsums...)

println(dfsums_all)

scans = collect(dfsums_all.scan)
times = collect(dfsums_all.time)
mjd = collect(dfsums_all.mjd)
rel_times = collect(dfsums_all.rel_time)

#ridge_plot2(chainpaths, scans, times)

csv_files = glob("scan*-skychain.csv", (joinpath((@__DIR__), "../CfA-Proj/stacking/csv-chains-grmhdtest3599/")))
#csv_files = glob("scan*-skychain.csv", (joinpath((@__DIR__), "../CfA-Proj/stacking/csv-chains-SgrA/")))


#ridge_plot_csv(csv_files, scans, times; field=:a, output="ridge_a_from_csv.png")
#ridge_plot_csv(csv_files, scans, times; field=:incl, output="ridge_incl_from_csv.png")



ridge_plot_csv_4(csv_files, scans, times; lims=(-1,1), xlabel=L"$a_{\mathrm{*}}$", field=:a, output="ridge_a.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(0, π/2), xlabel=L"$\theta_{\mathrm{o}}$", field=:incl, output="ridge_incl.png", utc=true, angular=false)
#ridge_plot_csv_4(csv_files, scans, times; lims=(0, 180), xlabel=L"$\theta_o$", field=:incl, output="ridge_incl_degrees.png", utc=true, angular=true)
ridge_plot_csv_4(csv_files, scans, times; lims=(0, 2π), xlabel=L"p.a.", field=:pa, output="ridge_pa.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(0, 10), xlabel=L"R_{peak}", field=:rpeak, output="ridge_rpeak.png", utc=true, angular=false)

ridge_plot_csv_4(csv_files, scans, times; lims=(40 / 180 * π, π / 2), xlabel="θs", field=:θs, output="ridge_θs.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(0.1, 10.0), xlabel="p1", field=:p1, output="ridge_p1.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(1.0, 10.0), xlabel="p2", field=:p2, output="ridge_p2.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(-π, π), xlabel="χ", field=:χ, output="ridge_χ.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(0.0, π/2), xlabel="ι", field=:ι, output="ridge_ι.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(0.0, 0.99), xlabel="βv", field=:βv, output="ridge_βv.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(-1.0, 5.0), xlabel="spec", field=:spec, output="ridge_spec.png", utc=true, angular=false)
ridge_plot_csv_4(csv_files, scans, times; lims=(-π, π), xlabel="η", field=:η, output="ridge_η.png", utc=true, angular=false)


#ridge_plot_csv_4(csv_files, scans, times; lims=(1, 89), xlabel=L"$\theta_o$", field=:incl, output="ridge_incl.png", utc=true, angular=true)

#ridge_plot_gaussians(csv_files, scans, times;
#    lims=(-1, 1),
#    xlabel=L"$a_*$", field=:a, output="ridge_a_with_gaussians.png", utc=true, angular=false,
#    gaussians=[(0.4, 0.8), (0.2, 0.5)]
#)

#=
df_global = CSV.read("stacker-chains/stacker_chain_ha_trunc_no-restrict_14_data.csv", DataFrame)
means, std = construct_meanstd(df_global)

a_gaussians = []

for idx in 1:1000:nrow(means)
    df_paramvals = DataFrame()

    meanval = means[!, "a"][idx]
    stdval = std[!, "a"][idx]

    meanval = Float64(meanval)
    stdval = Float64(stdval)

    gaussian_sample = (meanval, stdval)
    push!(a_gaussians, gaussian_sample)
end

incl_gaussians = []

for idx in 1:1000:nrow(means)
    df_paramvals = DataFrame()

    meanval = means[!, "incl"][idx]
    stdval = std[!, "incl"][idx]

    meanval = Float64(meanval)
    stdval = Float64(stdval)

    gaussian_sample = (meanval, stdval)
    push!(incl_gaussians, gaussian_sample)
end
=#

#ridgeplot_gaussians(csv_files, scans, times;
#    lims=(-1, 1),
#    xlabel=L"$a_*$", field=:a, output="ridge_a_with_gaussians.png", utc=true, angular=false,
   # gaussians=[(0.4, 0.8), (0.2, 0.5)]
#    gaussians=a_gaussians
#)

#ridgeplot_gaussians(csv_files, scans, times;
#    lims=(0, π/2),
#    xlabel=L"$\theta_o$", field=:incl, output="ridge_incl_with_gaussians.png", utc=true, angular=false,
   # gaussians=[(0.4, 0.8), (0.2, 0.5)]
#    gaussians=incl_gaussians
#)


#ridge_plot_with_gaussians(csv_files, scans, times; gaussians=[], subsample=10, outpath="ridgeplot_a.png")

#ridge_plot_with_scans_and_times(
#    csv_files, 
#    scans, 
#    times; 
#    field=:a, 
#    xlim=(-1,1), 
#    gaussians=[(-0.5, 0.1, 1.0), (0.3, 0.05, 0.8)],
#    subsample=10,
#    outpath="ridge_a_truncated.png",
#    xlabel=L"Spin Parameter $a$"
#)


#ridge_plot_expanded_2(
#    csv_files,         # e.g. ["scan1.csv", "scan2.csv", ...]
#    scans,            # e.g. [1, 2, 3, ...]
#    times;            # e.g. [60234.12, 60234.18, 60234.24, ...]
#
#    field = :a,
#    output = "ridge_expand_a.png",
#
#    # Posterior color (single or array)
#    posterior_color = (:steelblue, 0.6),
#
#    # Add Gaussian overlays (unit peak height)
#    gaussians = [
#        (μ = 50.0, σ = 5.0),
#        (μ = 70.0, σ = 3.0)
#    ],
#
#    # Truncate Gaussians and posterior densities to visible x-range
#    xlims = (-1, 1),
#
#    # LaTeX labels and title
#    xlabel = L"a_*",
#    ylabel = L"\text{Time (MJD)}",
#    title  = L"\text{Posterior Ridge Plot}",
#
#    # Choose type of y-axis
#    yaxis = :utc    # or :scan or :utc
#)
